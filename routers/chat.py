from fastapi import APIRouter, HTTPException
from fastapi.websockets import WebSocket
from sqlmodel import Session
from database import engine
from models import Room, Rooms, RoomSettings, RoomSettingsCreate
from services.voice_chat import VoiceChat
    
chat = APIRouter(tags=["Chat Room"])

voice_chat = VoiceChat()

@chat.post("/v1/create_room")
async def create_room(room: Room):
    res = await voice_chat.create_room(room.room_name)
    with Session(engine) as session:
        item = Rooms(room_name=room.room_name)
        session.add(item)
        session.commit()
    return res

# ========== 방별 설정 API ==========
@chat.get("/v1/room/{room_name}/settings")
async def get_room_settings(room_name: str):
    """특정 방의 설정 조회"""
    try:
        settings = voice_chat.get_room_settings(room_name)
        return {
            "room_name": room_name,
            "settings": settings
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"설정 조회 실패: {str(e)}")

@chat.put("/v1/room/{room_name}/settings")
async def update_room_settings(room_name: str, settings: RoomSettingsCreate):
    """특정 방의 설정 업데이트"""
    try:
        # 방 이름 일치 확인
        if settings.room_name != room_name:
            raise HTTPException(status_code=400, detail="URL과 요청 데이터의 방 이름이 일치하지 않습니다.")
        
        # 설정 업데이트
        update_data = settings.dict()
        update_data.pop('room_name')  # room_name 제거
        
        success = voice_chat.update_room_settings(room_name, **update_data)
        if success:
            return {
                "message": f"방 '{room_name}' 설정이 업데이트되었습니다.",
                "updated_settings": update_data
            }
        else:
            raise HTTPException(status_code=500, detail="설정 업데이트 실패")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"설정 업데이트 실패: {str(e)}")

@chat.post("/v1/room/{room_name}/settings/reset")
async def reset_room_settings(room_name: str):
    """특정 방의 설정을 기본값으로 리셋"""
    try:
        # 캐시에서 제거
        if room_name in voice_chat.room_settings_cache:
            del voice_chat.room_settings_cache[room_name]
        
        # 기본 설정으로 재생성
        settings = voice_chat.create_default_room_settings(room_name)
        
        return {
            "message": f"방 '{room_name}' 설정이 기본값으로 리셋되었습니다.",
            "default_settings": settings
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"설정 리셋 실패: {str(e)}")

@chat.get("/v1/room/{room_name}/settings/{setting_name}")
async def get_room_setting(room_name: str, setting_name: str):
    """특정 방의 특정 설정값 조회"""
    try:
        value = voice_chat.get_room_setting(room_name, setting_name)
        if value is None:
            raise HTTPException(status_code=404, detail=f"설정 '{setting_name}'를 찾을 수 없습니다.")
        
        return {
            "room_name": room_name,
            "setting_name": setting_name,
            "value": value
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"설정 조회 실패: {str(e)}")

@chat.put("/v1/room/{room_name}/settings/{setting_name}")
async def update_room_setting(room_name: str, setting_name: str, value: bool):
    """특정 방의 특정 설정값 업데이트"""
    try:
        # 유효한 설정인지 확인
        valid_settings = ['use_voice_enhance', 'hear_me', 'record_audio', 'classify_event', 'do_stt']
        if setting_name not in valid_settings:
            raise HTTPException(status_code=400, detail=f"유효하지 않은 설정: {setting_name}")
        
        # enhance_volume은 int 타입
        if setting_name == 'enhance_volume':
            try:
                value = int(value)
            except:
                raise HTTPException(status_code=400, detail="enhance_volume은 정수여야 합니다.")
        
        success = voice_chat.update_room_settings(room_name, **{setting_name: value})
        if success:
            return {
                "message": f"방 '{room_name}'의 '{setting_name}' 설정이 업데이트되었습니다.",
                "room_name": room_name,
                "setting_name": setting_name,
                "new_value": value
            }
        else:
            raise HTTPException(status_code=500, detail="설정 업데이트 실패")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"설정 업데이트 실패: {str(e)}")

# ========== 기존 전역 설정 API (호환성 유지) ==========

@chat.delete("/v1/room/{id}")
async def delete_room(id: int):
    res = await voice_chat.delete_room(id)
    if res is None:
        raise HTTPException(status_code=400, detail=f"Room id {id} is not found")
    return res

@chat.get("/v1/option_settings")
async def get_options():
    return {
        "hear_me": voice_chat.hear_me,
        "noise_remove": voice_chat.use_voice_enhance, 
        "do_stt": voice_chat.do_stt,
        "classify_event": voice_chat.classify_event,
        "record_audio": voice_chat.record_audio,
        "keep_test_room": voice_chat.keep_test_room
    }

# 기존 전역 설정 API들 (호환성 유지)
@chat.get("/v1/hear_me/{turn_on}")
async def hear_my_sound_default_false(turn_on: bool):
    voice_chat.hear_me = turn_on
    return {"message": f"Remove myself voice: {turn_on}"}

@chat.get("/v1/noise_remove/{turn_on}")
async def use_noise_remove_default_true(turn_on: bool):
    voice_chat.use_voice_enhance = turn_on
    return {"message": f"Use noise remove: {turn_on}"}

@chat.get("/v1/do_stt/{turn_on}")
async def do_stt_default_true(turn_on: bool):
    voice_chat.do_stt = turn_on
    return {"message": f"Do stt: {turn_on}"}

@chat.get("/v1/classify_event/{turn_on}")
async def classify_event_default_false(turn_on: bool):
    voice_chat.classify_event = turn_on
    return {"message": f"Classify event: {turn_on}"}

@chat.get("/v1/record_audio/{turn_on}")
async def record_audio_default_false(turn_on: bool):
    voice_chat.record_audio = turn_on
    return {"message": f"Record audio: {turn_on}"}

@chat.get("/v1/keep_test_room/{turn_on}")
async def keep_test_room_default_true(turn_on: bool):
    voice_chat.keep_test_room = turn_on
    return {"message": f"Keep test room: {turn_on}"}

# ========== 테스트 API ==========
@chat.post("/v1/test/start/{test_type}")
async def start_test(test_type: str):
    """테스트 모드 시작"""
    # 모든 테스트 플래그 초기화
    voice_chat.test_send_silence = False
    voice_chat.test_send_sine = False
    voice_chat.test_send_raw = False
    voice_chat.test_skip_enhance = False
    voice_chat.test_skip_mixing = False
    
    if test_type == "silence":
        voice_chat.test_send_silence = True
    elif test_type == "sine":
        voice_chat.test_send_sine = True
    elif test_type == "raw":
        voice_chat.test_send_raw = True
    elif test_type == "no_enhance":
        voice_chat.test_skip_enhance = True
    elif test_type == "no_mixing":
        voice_chat.test_skip_mixing = True
    elif test_type == "no_drop":
        voice_chat.test_disable_frame_drop = True
    elif test_type == "continuous_sine":
        voice_chat.test_continuous_sine = True
    elif test_type == "long_sine":
        voice_chat.test_long_sine = True
    elif test_type == "raw_mixed":
        voice_chat.test_raw_mixed = True
    elif test_type == "raw_mixed_enhanced":
        voice_chat.test_raw_mixed_enhanced = True
    elif test_type == "direct_send":
        voice_chat.test_direct_send = True
    elif test_type == "bypass_all":
        voice_chat.test_bypass_all = True
    elif test_type == "ultra_simple":
        voice_chat.test_ultra_simple = True
    elif test_type == "no_sync":
        voice_chat.test_no_sync = True
    elif test_type == "frame_512":
        voice_chat.test_frame_512 = True
    elif test_type == "frame_1024":
        voice_chat.test_frame_1024 = True
    elif test_type == "frame_2048":
        voice_chat.test_frame_2048 = True
    else:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 테스트 타입: {test_type}")
    
    voice_chat.reset_test_stats()
    return {
        "message": f"테스트 모드 시작: {test_type}",
        "note": "테스트는 '테스트' 방에서만 작동합니다",
        "room_name": "테스트"
    }

@chat.post("/v1/test/stop")
async def stop_test():
    """테스트 모드 중지"""
    voice_chat.test_send_silence = False
    voice_chat.test_send_sine = False
    voice_chat.test_send_raw = False
    voice_chat.test_skip_enhance = False
    voice_chat.test_skip_mixing = False
    voice_chat.test_disable_frame_drop = False
    voice_chat.test_continuous_sine = False
    voice_chat.test_long_sine = False
    voice_chat.test_raw_mixed = False
    voice_chat.test_raw_mixed_enhanced = False
    voice_chat.test_direct_send = False
    voice_chat.test_bypass_all = False
    voice_chat.test_ultra_simple = False
    voice_chat.test_no_sync = False
    voice_chat.test_frame_512 = False
    voice_chat.test_frame_1024 = False
    voice_chat.test_frame_2048 = False
    return {
        "message": "테스트 모드 중지",
        "note": "모든 방에서 정상 음성통신이 재개됩니다"
    }

@chat.get("/v1/test/stats")
async def get_test_stats():
    """테스트 통계 조회"""
    return voice_chat.get_test_stats()

@chat.post("/v1/test/reset")
async def reset_test_stats():
    """테스트 통계 초기화"""
    voice_chat.reset_test_stats()
    return {"message": "테스트 통계 초기화 완료"}

@chat.post("/v1/test/memory/start")
async def start_memory_monitoring():
    """메모리 모니터링 시작"""
    voice_chat.start_memory_monitoring()
    return {"message": "메모리 모니터링 시작"}

@chat.post("/v1/test/memory/stop")
async def stop_memory_monitoring():
    """메모리 모니터링 중지"""
    voice_chat.stop_memory_monitoring()
    return {"message": "메모리 모니터링 중지"}

@chat.post("/v1/test/cleanup")
async def cleanup_system():
    """시스템 정리 (버퍼 정리 + 가비지 컬렉션)"""
    voice_chat.cleanup_audio_buffers()
    voice_chat.force_garbage_collection()
    return {"message": "시스템 정리 완료"}

@chat.get("/v1/test/memory/status")
async def get_memory_status():
    """메모리 상태 조회"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # 오디오 버퍼 상태
        total_buffers = 0
        total_frames = 0
        for room_name, room_buffer in voice_chat.audio_buffers.items():
            for client_id, buffer in room_buffer.items():
                total_buffers += 1
                total_frames += len(buffer)
        
        return {
            "memory_mb": round(memory_mb, 1),
            "total_buffers": total_buffers,
            "total_frames": total_frames,
            "monitoring_enabled": voice_chat.memory_monitor_enabled
        }
    except ImportError:
        return {"error": "psutil이 설치되지 않았습니다"}
    except Exception as e:
        return {"error": str(e)}

@chat.websocket("/ws/room/{room_name}/{sr}/{dtype}/{device_id}")
async def websocket_endpoint(room_name: str, sr: int, dtype: str, device_id: str, websocket: WebSocket):
    await voice_chat.join_room(room_name, sr, dtype, device_id, websocket)
 