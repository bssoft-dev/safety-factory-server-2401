from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from starlette.middleware.cors import CORSMiddleware
from database import create_db_and_tables, engine, clear_room_data
from sqlmodel import Session, select
from models import Rooms, Events, Devices, SttRecords
from contextlib import asynccontextmanager
from utils.sys import aprint
from routers.log_mon import log_mon
from routers.chat import chat
from routers.simple_chat import simple_chat
from routers.playback_chat import playback_chat
from routers.bsm_integration import bsm_router
from routers.livekit_chat import router as livekit_router
import datetime
import os
import glob
from datetime import timedelta
from pydantic import BaseModel
from typing import List
from utils.sms import sms_send
from env import SMS_TEST_SEND_LIST

# SMS 요청 모델 정의
class SmsRequest(BaseModel):
    phone_numbers: List[str]
    content: str

class SmsTestRequest(BaseModel):
    content: str

@asynccontextmanager
async def lifespan(app:FastAPI):
    create_db_and_tables()
    clear_room_data()
    yield
    print("Yield done")
    
origins = [ "*" ]
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙 (웹 클라이언트용)
app.mount("/static", StaticFiles(directory="static"), name="static")


app.include_router(log_mon)
app.include_router(chat)
app.include_router(simple_chat)  # 간소화된 음성통신 라우터
app.include_router(playback_chat)  # WAV 파일 재생 라우터
app.include_router(bsm_router)  # BSM 연동 라우터 추가
app.include_router(livekit_router)  # LiveKit 기반 음성통신 라우터

@app.get("/v1/rooms")
async def get_info_rooms():
    with Session(engine) as session:
        return session.exec(select(Rooms)).all()


@app.get("/v1/events")
async def get_info_events():
    with Session(engine) as session:
        return session.exec(select(Events).order_by(Events.id.desc()).limit(20)).all()


@app.get("/")
async def read_root():
    return {"message": "다중 채팅방 지원 실시간 오디오 처리 서버"}

@app.get("/webclient", response_class=HTMLResponse)
async def get_web_client():
    """웹 클라이언트 페이지 제공"""
    try:
        with open("static/index.html", "r", encoding="utf-8") as file:
            html_content = file.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="웹 클라이언트 페이지를 찾을 수 없습니다.")

@app.get("/simple", response_class=HTMLResponse)
async def get_simple_web_client():
    """간단한 음성통신 웹 클라이언트 페이지 제공"""
    try:
        with open("static/simple.html", "r", encoding="utf-8") as file:
            html_content = file.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="간단한 웹 클라이언트 페이지를 찾을 수 없습니다.")

@app.get("/playback", response_class=HTMLResponse)
async def get_playback_web_client():
    """WAV 파일 재생 테스트 웹 클라이언트 페이지 제공"""
    try:
        with open("static/playback.html", "r", encoding="utf-8") as file:
            html_content = file.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="재생 테스트 웹 클라이언트 페이지를 찾을 수 없습니다.")

@app.post("/v1/warning")
async def post_warning(data: Events):
    with Session(engine) as session:
        query = select(Devices).where(Devices.device_id == data.device_id)
        device = session.exec(query).first()
        name = device.owner
    aprint(f"{data.time} warning: {name}, {data.event}", f"logs/warning.log")
    with Session(engine) as session:
        session.add(Events(device_id=data.device_id, event=data.event, time=data.time))
        session.commit()
    return {"message": data.event}

@app.get("/v1/stt_records/{room_name}")
async def get_stt_records(room_name: str):
    """특정 방의 STT 녹취록 조회"""
    with Session(engine) as session:
        records = session.exec(
            select(SttRecords)
            .where(SttRecords.room_name == room_name)
            .order_by(SttRecords.created_at.asc())
        ).all()
        return records

@app.get("/v1/stt_records/{room_name}/transcript")
async def get_transcript(room_name: str):
    """특정 방의 대화 녹취록을 시간순으로 정리"""
    with Session(engine) as session:
        records = session.exec(
            select(SttRecords)
            .where(SttRecords.room_name == room_name)
            .order_by(SttRecords.created_at.asc())
        ).all()
        
        transcript = []
        for record in records:
            transcript.append({
                "time": record.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "speaker": record.worker,
                "text": record.stt_text,
                "audio_file": record.wav_file_path
            })
        
        return {
            "room_name": room_name,
            "transcript": transcript,
            "total_records": len(transcript)
        }

@app.get("/v1/stt_records/{room_name}/with_audio")
async def get_stt_records_with_audio(room_name: str, start_date: str = None, end_date: str = None):
    """STT 기록과 음원 파일 정보 함께 조회 (원본/처리 후 모두)"""
    try:
        with Session(engine) as session:
            query = select(SttRecords).where(SttRecords.room_name == room_name)
            
            if start_date:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                query = query.where(SttRecords.created_at >= start_dt)
            
            if end_date:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
                query = query.where(SttRecords.created_at < end_dt)
            
            records = session.exec(query.order_by(SttRecords.created_at.asc())).all()
            
            result = []
            for record in records:
                # 원본/처리 후 음원 파일 경로 분리
                input_path = ""
                output_path = ""
                audio_exists = False
                
                if record.wav_file_path:
                    if "|" in record.wav_file_path:
                        # 원본과 처리 후 경로가 모두 있는 경우
                        paths = record.wav_file_path.split("|")
                        input_path = paths[0] if len(paths) > 0 else ""
                        output_path = paths[1] if len(paths) > 1 else ""
                    else:
                        # 하나의 경로만 있는 경우 (기존 데이터 호환성)
                        input_path = record.wav_file_path
                
                # 파일 존재 여부 확인
                if input_path:
                    full_input_path = os.path.join("recordings", input_path)
                    audio_exists = audio_exists or os.path.exists(full_input_path)
                
                if output_path:
                    full_output_path = os.path.join("recordings", output_path)
                    audio_exists = audio_exists or os.path.exists(full_output_path)
                
                result.append({
                    "id": record.id,
                    "room_name": record.room_name,
                    "worker": record.worker,
                    "stt_text": record.stt_text,
                    "input_wav_path": input_path,
                    "output_wav_path": output_path,
                    "audio_exists": audio_exists,
                    "created_at": record.created_at.strftime("%Y-%m-%d %H:%M:%S")
                })
            
            return {
                "room_name": room_name,
                "records": result,
                "total_records": len(result)
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"조회 오류: {str(e)}")

@app.get("/v1/audio_files/{room_name}")
async def get_audio_files(room_name: str, date: str = None, audio_type: str = "both"):
    """특정 방의 음원 파일 목록 조회 (원본/처리 후 모두)"""
    try:
        if date:
            target_date = datetime.strptime(date, '%Y-%m-%d')
        else:
            target_date = datetime.now()
        
        date_str = target_date.strftime('%Y-%m-%d')
        room_dir = os.path.join("recordings", date_str, room_name)
        
        if not os.path.exists(room_dir):
            return {"room_name": room_name, "date": date_str, "files": []}
        
        # 조회할 파일 타입 결정
        patterns = []
        if audio_type in ["both", "input"]:
            patterns.append(("input", os.path.join(room_dir, "input_*.wav")))
        if audio_type in ["both", "output"]:
            patterns.append(("output", os.path.join(room_dir, "output_*.wav")))
        
        audio_files = []
        file_mapping = {}
        
        for file_type, pattern in patterns:
            files = glob.glob(pattern)
            
            for file in files:
                filename = os.path.basename(file)
                try:
                    time_part = filename.split('_')[1]
                    worker = filename.split('_')[2].replace('.wav', '')
                    
                    hour = int(time_part.split('h')[0])
                    minute = int(time_part.split('h')[1].split('m')[0])
                    second = int(time_part.split('m')[1].split('s')[0])
                    
                    file_time = target_date.replace(hour=hour, minute=minute, second=second, microsecond=0)
                    
                    key = f"{time_part}_{worker}"
                    
                    if key not in file_mapping:
                        file_mapping[key] = {
                            "time": file_time.strftime("%H:%M:%S"),
                            "worker": worker,
                            "input_file": "",
                            "output_file": "",
                            "input_size": 0,
                            "output_size": 0
                        }
                    
                    file_mapping[key][f"{file_type}_file"] = os.path.relpath(file, "recordings")
                    file_mapping[key][f"{file_type}_size"] = os.path.getsize(file)
                    
                except (ValueError, IndexError):
                    continue
        
        # 결과 정리
        for key, file_info in file_mapping.items():
            audio_files.append(file_info)
        
        # 시간순 정렬
        audio_files.sort(key=lambda x: x['time'])
        
        return {
            "room_name": room_name,
            "date": date_str,
            "audio_type": audio_type,
            "files": audio_files,
            "total_files": len(audio_files)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"음원 파일 조회 오류: {str(e)}")

@app.get("/v1/audio_files/{room_name}/compare")
async def compare_audio_files(room_name: str, date: str = None):
    """원본과 처리 후 음원 파일 비교 정보"""
    try:
        if date:
            target_date = datetime.strptime(date, '%Y-%m-%d')
        else:
            target_date = datetime.now()
        
        date_str = target_date.strftime('%Y-%m-%d')
        room_dir = os.path.join("recordings", date_str, room_name)
        
        if not os.path.exists(room_dir):
            return {"room_name": room_name, "date": date_str, "comparison": []}
        
        input_pattern = os.path.join(room_dir, "input_*.wav")
        output_pattern = os.path.join(room_dir, "output_*.wav")
        
        input_files = glob.glob(input_pattern)
        output_files = glob.glob(output_pattern)
        
        comparison = []
        file_mapping = {}
        
        # input 파일들 처리
        for input_file in input_files:
            filename = os.path.basename(input_file)
            time_part = filename.split('_')[1]
            worker = filename.split('_')[2].replace('.wav', '')
            
            try:
                hour = int(time_part.split('h')[0])
                minute = int(time_part.split('h')[1].split('m')[0])
                second = int(time_part.split('m')[1].split('s')[0])
                
                file_time = target_date.replace(hour=hour, minute=minute, second=second, microsecond=0)
                
                key = f"{time_part}_{worker}"
                file_mapping[key] = {
                    "time": file_time.strftime("%H:%M:%S"),
                    "worker": worker,
                    "input_file": os.path.relpath(input_file, "recordings"),
                    "input_size": os.path.getsize(input_file),
                    "output_file": "",
                    "output_size": 0,
                    "has_both": False
                }
                
            except (ValueError, IndexError):
                continue
        
        # output 파일들 처리
        for output_file in output_files:
            filename = os.path.basename(output_file)
            time_part = filename.split('_')[1]
            worker = filename.split('_')[2].replace('.wav', '')
            
            try:
                hour = int(time_part.split('h')[0])
                minute = int(time_part.split('h')[1].split('m')[0])
                second = int(time_part.split('m')[1].split('s')[0])
                
                file_time = target_date.replace(hour=hour, minute=minute, second=second, microsecond=0)
                
                key = f"{time_part}_{worker}"
                if key in file_mapping:
                    file_mapping[key]["output_file"] = os.path.relpath(output_file, "recordings")
                    file_mapping[key]["output_size"] = os.path.getsize(output_file)
                    file_mapping[key]["has_both"] = True
                else:
                    file_mapping[key] = {
                        "time": file_time.strftime("%H:%M:%S"),
                        "worker": worker,
                        "input_file": "",
                        "input_size": 0,
                        "output_file": os.path.relpath(output_file, "recordings"),
                        "output_size": os.path.getsize(output_file),
                        "has_both": False
                    }
                
            except (ValueError, IndexError):
                continue
        
        # 결과 정리
        for key, file_info in file_mapping.items():
            comparison.append(file_info)
        
        # 시간순 정렬
        comparison.sort(key=lambda x: x['time'])
        
        # 통계 계산
        total_files = len(comparison)
        both_files = sum(1 for f in comparison if f['has_both'])
        input_only = sum(1 for f in comparison if f['input_file'] and not f['output_file'])
        output_only = sum(1 for f in comparison if f['output_file'] and not f['input_file'])
        
        return {
            "room_name": room_name,
            "date": date_str,
            "comparison": comparison,
            "statistics": {
                "total_files": total_files,
                "both_files": both_files,
                "input_only": input_only,
                "output_only": output_only
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"음원 파일 비교 오류: {str(e)}")

@app.post("/v1/sms/send", 
         summary="SMS 전송",
         description="지정된 전화번호 목록으로 SMS를 전송합니다.",
         response_description="SMS 전송 결과")
async def send_sms(request: SmsRequest):
    """
    SMS 전송 API
    
    **Parameters:**
    - **phone_numbers**: 수신자 전화번호 목록 (예: ["01012345678", "01087654321"])
    - **content**: 전송할 메시지 내용 (최대 90자)
    
    **Example Request:**
    ```json
    {
        "phone_numbers": ["01012345678", "01087654321"],
        "content": "안전관리 시스템 알림: 위험 상황이 감지되었습니다."
    }
    ```
    
    **Response:**
    ```json
    {
        "message": "SMS 전송 완료",
        "recipients": ["01012345678", "01087654321"],
        "content": "안전관리 시스템 알림: 위험 상황이 감지되었습니다.",
        "sent_at": "2025-01-27 14:30:00"
    }
    ```
    """
    try:
        # 전화번호 유효성 검사
        if not request.phone_numbers:
            raise HTTPException(status_code=400, detail="전화번호 목록이 비어있습니다.")
        
        # 메시지 내용 검사
        if not request.content or len(request.content.strip()) == 0:
            raise HTTPException(status_code=400, detail="메시지 내용이 비어있습니다.")
        
        if len(request.content) > 90:
            raise HTTPException(status_code=400, detail="메시지 내용이 90자를 초과합니다.")
        
        # SMS 전송
        await sms_send(request.phone_numbers, request.content)
        
        # 로그 기록
        aprint(f"SMS 전송 완료: {len(request.phone_numbers)}명, 내용: {request.content[:50]}...", "logs/sms.log")
        
        return {
            "message": "SMS 전송 완료",
            "recipients": request.phone_numbers,
            "content": request.content,
            "sent_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        aprint(f"SMS 전송 실패: {str(e)}", "logs/sms.log")
        raise HTTPException(status_code=500, detail=f"SMS 전송 실패: {str(e)}")

@app.post("/v1/sms/test", 
         summary="SMS 테스트 전송",
         description="테스트용 전화번호로 SMS를 전송합니다.",
         response_description="SMS 테스트 전송 결과")
async def send_test_sms(request: SmsTestRequest):
    """
    SMS 테스트 전송 API
    
    **Parameters:**
    - **content**: 전송할 메시지 내용 (최대 90자)
    
    **Example Request:**
    ```json
    {
        "content": "SMS 테스트 메시지입니다."
    }
    ```
    
    **Response:**
    ```json
    {
        "message": "SMS 테스트 전송 완료",
        "recipients": ["01051601747"],
        "content": "SMS 테스트 메시지입니다.",
        "sent_at": "2025-01-27 14:30:00"
    }
    ```
    """
    try:
        # 메시지 내용 검사
        if not request.content or len(request.content.strip()) == 0:
            raise HTTPException(status_code=400, detail="메시지 내용이 비어있습니다.")
        
        if len(request.content) > 90:
            raise HTTPException(status_code=400, detail="메시지 내용이 90자를 초과합니다.")
        
        # 테스트 전화번호로 SMS 전송
        await sms_send(SMS_TEST_SEND_LIST, request.content)
        
        # 로그 기록
        aprint(f"SMS 테스트 전송 완료: {len(SMS_TEST_SEND_LIST)}명, 내용: {request.content[:50]}...", "logs/sms.log")
        
        return {
            "message": "SMS 테스트 전송 완료",
            "recipients": SMS_TEST_SEND_LIST,
            "content": request.content,
            "sent_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        aprint(f"SMS 테스트 전송 실패: {str(e)}", "logs/sms.log")
        raise HTTPException(status_code=500, detail=f"SMS 테스트 전송 실패: {str(e)}")

@app.get("/v1/sms/config", 
         summary="SMS 설정 조회",
         description="현재 SMS 설정 정보를 조회합니다.",
         response_description="SMS 설정 정보")
async def get_sms_config():
    """
    SMS 설정 정보 조회 API
    
    **Response:**
    ```json
    {
        "service_id": "ncp:sms:kr:283078791279:smartbell",
        "sender_phone": "01022498703",
        "test_recipients": ["01051601747"],
        "status": "활성화"
    }
    ```
    """
    try:
        from env import SMS_SERVICE_ID, SMS_PHONE_NUMBER, SMS_TEST_SEND_LIST
        
        return {
            "service_id": SMS_SERVICE_ID,
            "sender_phone": SMS_PHONE_NUMBER,
            "test_recipients": SMS_TEST_SEND_LIST,
            "status": "활성화"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SMS 설정 조회 실패: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=24015, reload=True, log_config='utils/log_conf.json')
