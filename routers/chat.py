from fastapi import APIRouter, HTTPException
from fastapi.websockets import WebSocket
from sqlmodel import Session
from database import engine
from models import Room, Rooms
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

@chat.websocket("/ws/room/{room_name}/{sr}/{dtype}/{device_id}")
async def websocket_endpoint(room_name: str, sr: int, dtype: str, device_id: str, websocket: WebSocket):
    await voice_chat.join_room(room_name, sr, dtype, device_id, websocket)
 