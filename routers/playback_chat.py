from fastapi import APIRouter, HTTPException
from fastapi.websockets import WebSocket
from sqlmodel import Session
from database import engine
from models import Room, Rooms
from services.playback_voice_chat import PlaybackVoiceChat
    
playback_chat = APIRouter(tags=["Playback Chat Room"])

playback_voice_chat = PlaybackVoiceChat()

@playback_chat.post("/v1/playback/create_room")
async def create_room(room: Room):
    res = await playback_voice_chat.create_room(room.room_name)
    with Session(engine) as session:
        item = Rooms(room_name=room.room_name)
        session.add(item)
        session.commit()
    return res

@playback_chat.delete("/v1/playback/room/{id}")
async def delete_room(id: int):
    res = await playback_voice_chat.delete_room(id)
    if res is None:
        raise HTTPException(status_code=400, detail=f"Room id {id} is not found")
    return res

@playback_chat.get("/v1/playback/rooms")
async def get_rooms():
    return playback_voice_chat.get_rooms()

@playback_chat.websocket("/ws/playback/room/{room_name}/{sr}/{dtype}/{device_id}")
async def websocket_endpoint(room_name: str, sr: int, dtype: str, device_id: str, websocket: WebSocket):
    await playback_voice_chat.join_room(room_name, sr, dtype, device_id, websocket)
