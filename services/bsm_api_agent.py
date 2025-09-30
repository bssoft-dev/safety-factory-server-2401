import asyncio
import aiohttp
import json
from datetime import datetime
from typing import Optional, Dict, Any, Union
from enum import Enum
import os

class EventSeverity(Enum):
    CRITICAL = "critical"
    INFO = "info"

class EventName(Enum):
    SCREAM_DETECTED = "scream_detected"
    ALARM_DETECTED = "alarm_detected"
    SHOCK_DETECTED = "shock_detected"
    STT_RESULT = "stt_result"

class BSMApiAgent:
    def __init__(self):
        # BSM 서버 설정 - 환경변수에서 읽기
        self.bsm_base_url = os.getenv('BSM_URL', 'https://bsm.bs-soft.co.kr')
        self.bsm_api_key = os.getenv('BSM_API_KEY', '')
        self.safety_server_id = os.getenv('SAFETY_SERVER_ID', 'safety-server-main')
        
        # BSM 이벤트 수신 엔드포인트
        self.bsm_event_endpoint = f"{self.bsm_base_url}/api/v1/private/ingress/safety-server"
        
        # HTTP 클라이언트 헤더
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.bsm_api_key}' if self.bsm_api_key else '',
            'User-Agent': 'Safety-Server-BSM-Agent/1.0'
        }
        
        # 재시도 설정
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # 초기화 완료를 콘솔에 출력 (비동기 로깅 대신)
        print(f"BSM API Agent initialized - Target: {self.bsm_base_url}")

    async def report_audio_event(self, 
                                device_id: str, 
                                room_name: str, 
                                event_name: EventName, 
                                severity: EventSeverity,
                                payload: Dict[str, Any],
                                occurred_at: Optional[datetime] = None) -> bool:
        """
        BSM에 음성 분석 이벤트를 보고합니다.
        
        Args:
            device_id: 음원을 보낸 디바이스 ID
            room_name: safety-server 방 이름
            event_name: 이벤트 종류
            severity: 심각도
            payload: 상세 데이터
            occurred_at: 발생 시각 (None이면 현재 시각)
        
        Returns:
            bool: 성공 여부
        """
        if not occurred_at:
            occurred_at = datetime.now()
        
        # BSM 요청 본문 구성
        request_data = {
            "device_id": device_id,
            "room_name": room_name,
            "event_type": "audio_analysis",
            "event_name": event_name.value,
            "severity": severity.value,
            "payload": payload,
            "occurred_at": occurred_at.isoformat()
        }
        
        # BSM에 이벤트 보고
        success = await self._send_to_bsm(request_data)
        
        if success:
            # 성공 시 비동기 로깅 (안전하게)
            try:
                from utils.sys import aprint
                aprint(f"BSM 이벤트 보고 성공: {device_id} - {event_name.value}")
            except:
                print(f"BSM 이벤트 보고 성공: {device_id} - {event_name.value}")
        else:
            # 실패 시 비동기 로깅 (안전하게)
            try:
                from utils.sys import aprint
                aprint(f"BSM 이벤트 보고 실패: {device_id} - {event_name.value}")
            except:
                print(f"BSM 이벤트 보고 실패: {device_id} - {event_name.value}")
        
        return success

    async def report_scream_event(self, device_id: str, room_name: str, confidence: float, audio_file_path: str = "") -> bool:
        """비명 감지 이벤트를 BSM에 보고"""
        payload = {
            "detection_type": "scream",
            "confidence": confidence,
            "audio_file_path": audio_file_path,
            "description": f"비명 소리가 감지되었습니다 (신뢰도: {confidence:.2f})"
        }
        
        return await self.report_audio_event(
            device_id=device_id,
            room_name=room_name,
            event_name=EventName.SCREAM_DETECTED,
            severity=EventSeverity.CRITICAL,
            payload=payload
        )

    async def report_alarm_event(self, device_id: str, room_name: str, confidence: float, audio_file_path: str = "") -> bool:
        """경보음 감지 이벤트를 BSM에 보고"""
        payload = {
            "detection_type": "alarm",
            "confidence": confidence,
            "audio_file_path": audio_file_path,
            "description": f"경보음이 감지되었습니다 (신뢰도: {confidence:.2f})"
        }
        
        return await self.report_audio_event(
            device_id=device_id,
            room_name=room_name,
            event_name=EventName.ALARM_DETECTED,
            severity=EventSeverity.CRITICAL,
            payload=payload
        )

    async def report_shock_event(self, device_id: str, room_name: str, confidence: float, audio_file_path: str = "") -> bool:
        """충격/깨짐 소리 감지 이벤트를 BSM에 보고"""
        payload = {
            "detection_type": "shock",
            "confidence": confidence,
            "audio_file_path": audio_file_path,
            "description": f"충격/깨짐 소리가 감지되었습니다 (신뢰도: {confidence:.2f})"
        }
        
        return await self.report_audio_event(
            device_id=device_id,
            room_name=room_name,
            event_name=EventName.SHOCK_DETECTED,
            severity=EventSeverity.CRITICAL,
            payload=payload
        )

    async def report_stt_result(self, device_id: str, room_name: str, stt_text: str, 
                              worker: str = "unknown", 
                              input_wav_path: str = "", 
                              output_wav_path: str = "") -> bool:
        """STT 결과를 BSM에 보고"""
        payload = {
            "stt_text": stt_text,
            "worker": worker,
            "input_wav_path": input_wav_path,
            "output_wav_path": output_wav_path,
            "description": f"음성 인식 결과: {stt_text}"
        }
        
        return await self.report_audio_event(
            device_id=device_id,
            room_name=room_name,
            event_name=EventName.STT_RESULT,
            severity=EventSeverity.INFO,
            payload=payload
        )

    async def health_check(self) -> bool:
        """BSM 서버의 상태를 확인"""
        try:
            async with aiohttp.ClientSession() as session:
                health_url = f"{self.bsm_base_url}/api/v1/health"
                async with session.get(health_url, headers=self.headers, timeout=5) as response:
                    return response.status == 200
        except Exception as e:
            print(f"BSM 헬스체크 실패: {e}")
            return False

    async def get_active_devices(self) -> list:
        """BSM에서 현재 활성화된 디바이스 목록을 가져옴 (선택적 기능)"""
        try:
            async with aiohttp.ClientSession() as session:
                devices_url = f"{self.bsm_base_url}/api/v1/devices/active"
                async with session.get(devices_url, headers=self.headers, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('devices', [])
                    else:
                        print(f"BSM 디바이스 목록 조회 실패: {response.status}")
                        return []
        except Exception as e:
            print(f"BSM 디바이스 목록 조회 오류: {e}")
            return []

    async def report_room_status(self, room_name: str, active_clients: int, features: dict) -> bool:
        """safety-server 방 상태를 BSM에 보고 (선택적 기능)"""
        try:
            status_data = {
                "room_name": room_name,
                "active_clients": active_clients,
                "features": features,
                "timestamp": datetime.now().isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                status_url = f"{self.bsm_base_url}/api/v1/private/safety-server/room-status"
                async with session.post(status_url, 
                                      json=status_data, 
                                      headers=self.headers, 
                                      timeout=5) as response:
                    return response.status == 202
        except Exception as e:
            print(f"BSM 방 상태 보고 오류: {e}")
            return False

    async def _send_to_bsm(self, data: dict) -> bool:
        """BSM 서버에 데이터를 전송 (재시도 로직 포함)"""
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.bsm_event_endpoint, 
                                          json=data, 
                                          headers=self.headers, 
                                          timeout=10) as response:
                        if response.status == 202:  # BSM이 요청을 수락
                            return True
                        elif response.status == 401:
                            print("BSM API 인증 실패 - API 키를 확인하세요")
                            return False
                        elif response.status == 400:
                            error_text = await response.text()
                            print(f"BSM API 요청 형식 오류: {error_text}")
                            return False
                        else:
                            print(f"BSM API 응답 오류: {response.status}")
                            
            except asyncio.TimeoutError:
                print(f"BSM API 타임아웃 (시도 {attempt + 1}/{self.max_retries})")
            except Exception as e:
                print(f"BSM API 통신 오류 (시도 {attempt + 1}/{self.max_retries}): {e}")
            
            # 마지막 시도가 아니면 잠시 대기 후 재시도
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        return False

    def extract_device_id_from_room(self, room_name: str) -> str:
        """
        방 이름에서 디바이스 ID를 추출
        BSM이 생성하는 방 이름 패턴: bsm-room-{device_id}-{uuid}
        또는 기존 방식과의 호환성을 위해 client_info에서 추출
        """
        if room_name.startswith("bsm-room-"):
            # BSM 패턴: bsm-room-{device_id}-{uuid}
            parts = room_name.split("-")
            if len(parts) >= 3:
                return parts[2]  # device_id 부분
        
        # 기본값 또는 레거시 처리
        return "unknown"

# 지연 초기화를 위한 전역 변수
_bsm_agent_instance = None

def get_bsm_agent() -> BSMApiAgent:
    """BSM API 에이전트의 싱글톤 인스턴스를 반환"""
    global _bsm_agent_instance
    if _bsm_agent_instance is None:
        _bsm_agent_instance = BSMApiAgent()
    return _bsm_agent_instance

# 호환성을 위한 전역 변수 (지연 초기화)
bsm_agent = get_bsm_agent()
