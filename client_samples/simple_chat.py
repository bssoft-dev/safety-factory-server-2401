from fastapi import APIRouter, HTTPException
from fastapi.websockets import WebSocket
from sqlmodel import Session
from database import engine
from models import Room, Rooms
from services.simple_voice_chat import SimpleVoiceChat
    
simple_chat = APIRouter(tags=["Simple Chat Room"])

simple_voice_chat = SimpleVoiceChat()

@simple_chat.post("/v1/simple/create_room")
async def create_room(room: Room):
    res = await simple_voice_chat.create_room(room.room_name)
    with Session(engine) as session:
        item = Rooms(room_name=room.room_name)
        session.add(item)
        session.commit()
    return res

@simple_chat.delete("/v1/simple/room/{id}")
async def delete_room(id: int):
    res = await simple_voice_chat.delete_room(id)
    if res is None:
        raise HTTPException(status_code=400, detail=f"Room id {id} is not found")
    return res

@simple_chat.get("/v1/simple/rooms")
async def get_rooms():
    return simple_voice_chat.get_rooms()

@simple_chat.websocket("/ws/simple/room/{room_name}/{sr}/{dtype}/{device_id}")
async def websocket_endpoint(room_name: str, sr: int, dtype: str, device_id: str, websocket: WebSocket):
    await simple_voice_chat.join_room(room_name, sr, dtype, device_id, websocket)
