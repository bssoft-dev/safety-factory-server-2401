from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
from datetime import datetime
from services.bsm_api_agent import bsm_agent
from services.voice_chat import VoiceChat
from models import Room
from utils.sys import aprint

# BSM 연동 전용 라우터
bsm_router = APIRouter(prefix="/api/v1/bsm", tags=["BSM Integration"])

# 보안을 위한 Bearer 토큰 검증
security = HTTPBearer()

# 허용된 IP 주소 (BSM 서버 IP) - 환경변수에서 읽기
ALLOWED_BSM_IPS = os.getenv('ALLOWED_BSM_IPS', '').split(',') if os.getenv('ALLOWED_BSM_IPS') else []
BSM_API_KEY = os.getenv('BSM_API_KEY', '')

class BSMRoomRequest(BaseModel):
    room_name: str
    device_id: str
    audio_settings: Optional[Dict[str, Any]] = None

class BSMStatusResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

def verify_bsm_access(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """BSM 접근 권한 검증"""
    client_ip = request.client.host
    
    # IP 화이트리스트 체크 (설정된 경우)
    if ALLOWED_BSM_IPS and client_ip not in ALLOWED_BSM_IPS:
        raise HTTPException(status_code=403, detail=f"IP {client_ip} not allowed")
    
    # API 키 검증 (설정된 경우)
    if BSM_API_KEY and credentials.credentials != BSM_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True

# VoiceChat 인스턴스 (기존 것과 동일)
voice_chat = VoiceChat()

@bsm_router.post("/create-room", response_model=BSMStatusResponse)
async def create_room_for_bsm(
    request: BSMRoomRequest, 
    _: bool = Depends(verify_bsm_access)
):
    """
    BSM에서 호출하는 방 생성 API
    보안이 강화된 버전으로, BSM만 호출 가능
    """
    try:
        # 방 이름 검증 (BSM 패턴 확인)
        if not request.room_name.startswith("bsm-room-"):
            raise HTTPException(status_code=400, detail="Invalid room name pattern")
        
        # 기존 voice_chat의 create_room 메서드 사용
        result = await voice_chat.create_room(request.room_name)
        
        # WebSocket URL 생성
        websocket_url = f"ws://safety-server.bs-soft.co.kr:24015/ws/room/{request.room_name}/16000/int16/{request.device_id}"
        
        aprint(f"BSM에서 방 생성 요청: {request.room_name} for device {request.device_id}")
        
        return BSMStatusResponse(
            status="success",
            message=f"Room '{request.room_name}' created successfully",
            data={
                "room_name": request.room_name,
                "websocket_url": websocket_url,
                "device_id": request.device_id,
                "audio_settings": request.audio_settings
            }
        )
    except Exception as e:
        aprint(f"BSM 방 생성 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@bsm_router.delete("/room/{room_name}")
async def delete_room_for_bsm(
    room_name: str,
    _: bool = Depends(verify_bsm_access)
):
    """BSM에서 호출하는 방 삭제 API"""
    try:
        # room_name으로 room ID 찾기
        from database import engine
        from sqlmodel import Session, select
        from models import Rooms
        
        with Session(engine) as session:
            room = session.exec(select(Rooms).filter(Rooms.room_name == room_name)).first()
            if not room:
                raise HTTPException(status_code=404, detail=f"Room '{room_name}' not found")
            
            # 기존 delete_room 메서드 사용
            result = await voice_chat.delete_room(room.id)
            
            aprint(f"BSM에서 방 삭제 요청: {room_name}")
            
            return BSMStatusResponse(
                status="success",
                message=f"Room '{room_name}' deleted successfully"
            )
    except HTTPException:
        raise
    except Exception as e:
        aprint(f"BSM 방 삭제 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@bsm_router.get("/health")
async def health_check():
    """BSM에서 safety-server 상태를 확인하는 헬스체크 API"""
    try:
        # 기본 서비스 상태 확인
        room_count = len(voice_chat.rooms)
        total_clients = sum(len(clients) for clients in voice_chat.rooms.values())
        
        # 옵션 설정 상태
        options = {
            "hear_me": voice_chat.hear_me,
            "noise_remove": voice_chat.use_voice_enhance,
            "do_stt": voice_chat.do_stt,
            "classify_event": voice_chat.classify_event,
            "record_audio": voice_chat.record_audio
        }
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "rooms": room_count,
            "total_clients": total_clients,
            "options": options,
            "version": "1.0.0"
        }
    except Exception as e:
        aprint(f"헬스체크 오류: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

@bsm_router.get("/rooms/status")
async def get_rooms_status(_: bool = Depends(verify_bsm_access)):
    """BSM에서 현재 활성 방들의 상태를 조회하는 API"""
    try:
        rooms_status = []
        
        for room_name, clients in voice_chat.rooms.items():
            client_count = len(clients)
            client_info = []
            
            # 각 클라이언트 정보 수집
            for client_ws in clients:
                client_id = id(client_ws)
                if client_id in voice_chat.client_info:
                    info = voice_chat.client_info[client_id]
                    client_info.append({
                        "person_name": info.get("person_name", "unknown"),
                        "dtype": info.get("dtype", "unknown"),
                        "sr": info.get("sr", 0),
                        "is_webbrowser": info.get("is_webbrowser", False)
                    })
            
            rooms_status.append({
                "room_name": room_name,
                "client_count": client_count,
                "clients": client_info,
                "device_id": bsm_agent.extract_device_id_from_room(room_name)
            })
        
        return {
            "status": "success",
            "total_rooms": len(rooms_status),
            "rooms": rooms_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        aprint(f"방 상태 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@bsm_router.post("/test-connection")
async def test_bsm_connection(_: bool = Depends(verify_bsm_access)):
    """BSM과 safety-server 간 연결 테스트"""
    try:
        # BSM 헬스체크 수행
        bsm_healthy = await bsm_agent.health_check()
        
        # 테스트 이벤트 전송 (실제 이벤트가 아님을 명시)
        test_result = await bsm_agent.report_audio_event(
            device_id="test-device",
            room_name="test-room",
            event_name=bsm_agent.EventName.STT_RESULT,
            severity=bsm_agent.EventSeverity.INFO,
            payload={
                "test": True,
                "message": "Connection test from safety-server",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return {
            "status": "success",
            "bsm_reachable": bsm_healthy,
            "event_delivery": test_result,
            "message": "Connection test completed"
        }
    except Exception as e:
        aprint(f"BSM 연결 테스트 오류: {e}")
        return {
            "status": "error",
            "bsm_reachable": False,
            "event_delivery": False,
            "message": str(e)
        }

@bsm_router.get("/config")
async def get_bsm_config():
    """현재 BSM 연동 설정 정보 조회 (민감하지 않은 정보만)"""
    return {
        "bsm_url": bsm_agent.bsm_base_url,
        "safety_server_id": bsm_agent.safety_server_id,
        "has_api_key": bool(bsm_agent.bsm_api_key),
        "allowed_ips": ALLOWED_BSM_IPS if ALLOWED_BSM_IPS else ["any"],
        "agent_version": "1.0.0"
    } 