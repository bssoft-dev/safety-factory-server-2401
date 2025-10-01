from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import asyncio

from services.livekit_voice_chat import LiveKitVoiceChat
from utils.sys import aprint

# 라우터 생성
router = APIRouter(prefix="/api/livekit", tags=["LiveKit Voice Chat"])

# LiveKit 서비스 인스턴스 (실제 운영 서버 URL 사용)
livekit_service = LiveKitVoiceChat(
    livekit_url="wss://livekit.bs-soft.co.kr",
    api_key="devkey",
    api_secret="secret"
)

# Pydantic 모델들
class RoomCreateRequest(BaseModel):
    room_name: str

class RoomJoinRequest(BaseModel):
    room_name: str
    participant_name: str
    participant_identity: Optional[str] = None
    device_id: Optional[str] = None

class RoomDeleteRequest(BaseModel):
    room_id: int

class RoomSettingsUpdateRequest(BaseModel):
    room_name: str
    use_voice_enhance: Optional[bool] = None
    hear_me: Optional[bool] = None
    record_audio: Optional[bool] = None
    classify_event: Optional[bool] = None
    do_stt: Optional[bool] = None
    enhance_volume: Optional[int] = None

class ParticipantEnhancementRequest(BaseModel):
    participant_identity: str
    enabled: bool = True
    enhancement_type: str = "light"  # "light" or "full"

# API 엔드포인트들

@router.get("/rooms")
async def get_rooms():
    """방 목록 조회"""
    try:
        rooms = await livekit_service.get_rooms()
        return {"rooms": rooms}
    except Exception as e:
        aprint(f"방 목록 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get rooms: {e}")

@router.post("/rooms")
async def create_room(request: RoomCreateRequest):
    """방 생성"""
    try:
        result = await livekit_service.create_room(request.room_name)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        aprint(f"방 생성 오류: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create room: {e}")

@router.delete("/rooms")
async def delete_room(request: RoomDeleteRequest):
    """방 삭제"""
    try:
        result = await livekit_service.delete_room(request.room_id)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        aprint(f"방 삭제 오류: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete room: {e}")

@router.post("/join")
async def join_room(request: RoomJoinRequest):
    """방 참가"""
    try:
        result = await livekit_service.join_room(
            room_name=request.room_name,
            participant_name=request.participant_name,
            participant_identity=request.participant_identity,
            device_id=request.device_id
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # 방 핸들러 설정
        await livekit_service.setup_room_handlers(request.room_name)
        
        return result
    except Exception as e:
        aprint(f"방 참가 오류: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to join room: {e}")

@router.get("/rooms/{room_name}/participants")
async def get_room_participants(room_name: str):
    """방 참가자 목록 조회"""
    try:
        participants = await livekit_service.get_room_participants(room_name)
        return {"participants": participants}
    except Exception as e:
        aprint(f"참가자 목록 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get participants: {e}")

@router.get("/rooms/{room_name}/settings")
async def get_room_settings(room_name: str):
    """방 설정 조회"""
    try:
        settings = livekit_service.get_room_settings(room_name)
        return {"settings": settings}
    except Exception as e:
        aprint(f"방 설정 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get room settings: {e}")

@router.put("/rooms/{room_name}/settings")
async def update_room_settings(room_name: str, request: RoomSettingsUpdateRequest):
    """방 설정 업데이트"""
    try:
        # 요청 데이터에서 None이 아닌 값들만 추출
        update_data = {}
        if request.use_voice_enhance is not None:
            update_data["use_voice_enhance"] = request.use_voice_enhance
        if request.hear_me is not None:
            update_data["hear_me"] = request.hear_me
        if request.record_audio is not None:
            update_data["record_audio"] = request.record_audio
        if request.classify_event is not None:
            update_data["classify_event"] = request.classify_event
        if request.do_stt is not None:
            update_data["do_stt"] = request.do_stt
        if request.enhance_volume is not None:
            update_data["enhance_volume"] = request.enhance_volume
        
        success = livekit_service.update_room_settings(room_name, **update_data)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to update room settings")
        
        return {"message": "Room settings updated successfully", "settings": update_data}
    except Exception as e:
        aprint(f"방 설정 업데이트 오류: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update room settings: {e}")

@router.post("/token")
async def generate_token(request: RoomJoinRequest):
    """LiveKit 토큰 생성"""
    try:
        token = livekit_service.generate_token(
            room_name=request.room_name,
            participant_name=request.participant_name,
            participant_identity=request.participant_identity
        )
        
        if not token:
            raise HTTPException(status_code=400, detail="Failed to generate token")
        
        return {
            "token": token,
            "url": livekit_service.livekit_url,
            "room_name": request.room_name,
            "participant_name": request.participant_name,
            "participant_identity": request.participant_identity or request.participant_name
        }
    except Exception as e:
        aprint(f"토큰 생성 오류: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate token: {e}")

@router.get("/health")
async def health_check():
    """서비스 상태 확인"""
    try:
        return {
            "status": "healthy",
            "service": "LiveKit Voice Chat",
            "rooms_count": len(livekit_service.rooms),
            "livekit_url": livekit_service.livekit_url
        }
    except Exception as e:
        aprint(f"상태 확인 오류: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")

# ========== 소음제거 관련 API ==========

@router.post("/enhancement/set")
async def set_participant_enhancement(request: ParticipantEnhancementRequest):
    """참가자별 소음제거 설정"""
    try:
        # 입력 검증
        if request.enhancement_type not in ["light", "full"]:
            raise HTTPException(
                status_code=400, 
                detail="Invalid enhancement type. Use 'light' or 'full'"
            )
        
        result = livekit_service.set_participant_enhancement(
            participant_identity=request.participant_identity,
            enabled=request.enabled,
            enhancement_type=request.enhancement_type
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        aprint(f"소음제거 설정 오류: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set enhancement: {e}")

@router.get("/enhancement/status")
async def get_all_enhancement_status():
    """모든 참가자의 소음제거 설정 상태 조회"""
    try:
        status = {}
        for participant_id in livekit_service.participant_enhance_settings:
            status[participant_id] = livekit_service.get_participant_enhancement(participant_id)
        
        return {
            "total_participants": len(status),
            "enhancement_status": status
        }
        
    except Exception as e:
        aprint(f"전체 소음제거 상태 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get enhancement status: {e}")

@router.get("/enhancement/participant/{participant_identity}")
async def get_participant_enhancement(participant_identity: str):
    """참가자별 소음제거 설정 조회"""
    try:
        result = livekit_service.get_participant_enhancement(participant_identity)
        return result
        
    except Exception as e:
        aprint(f"소음제거 설정 조회 오류: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get enhancement: {e}")

# 서비스 종료 시 정리
@router.on_event("shutdown")
async def shutdown_event():
    """서비스 종료 시 리소스 정리"""
    try:
        await livekit_service.cleanup()
        aprint("LiveKit 서비스 정리 완료")
    except Exception as e:
        aprint(f"서비스 정리 오류: {e}")
