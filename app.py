from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from audio import AudioProcessor
from database import create_db_and_tables, engine
from sqlmodel import Session, select
from models import Rooms
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app:FastAPI):
    create_db_and_tables()
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


class Room(BaseModel):
    room_name: str

audio_processor = AudioProcessor()

@app.get("/v1/rooms")
async def get_info_rooms():
    return audio_processor.get_rooms()

@app.post("/v1/create_room")
async def create_room(room: Room):
    res = await audio_processor.create_room(room.room_name)
    with Session(engine) as session:
        item = Rooms(room_name=room.room_name)
        session.add(item)
        session.commit()
    return res

@app.get("/v1/room/{room_name}/add_person")
async def add_person_to_room(room_name: str):
    # audio_processor.rooms_num_person[room_name] += 1
    with Session(engine) as session:
        room = session.exec(select(Rooms).filter(Rooms.room_name == room_name)).first()
        room.num_person = room.num_person + 1
        session.add(room)
        session.commit()
    return {"message": f"Person added to room '{room_name}'"}

@app.get("/v1/room/{room_name}/remove_person")
async def remove_person_from_room(room_name: str):
    # audio_processor.rooms_num_person[room_name] -= 1
    with Session(engine) as session:
        room = session.exec(select(Rooms).filter(Rooms.room_name == room_name)).first()
        room.num_person = room.num_person - 1
        session.add(room)
        session.commit()
    return {"message": f"Person removed from room '{room_name}'"}

@app.delete("/v1/room/{room_name}")
async def delete_room(room_name: str):
    if room_name in audio_processor.rooms:
        for ws in audio_processor.rooms[room_name]:
            await ws.close(code=4004)
        del audio_processor.rooms[room_name]
        del audio_processor.audio_buffers[room_name]
    with Session(engine) as session:
        room = session.exec(select(Rooms).filter(Rooms.room_name == room_name)).first()
        session.delete(room)
        session.commit()
    return {"message": f"Room '{room_name}' deleted successfully"}

@app.get("/v1/noise_remove/{turn_on}")
async def use_noise_remove(turn_on: bool):
    audio_processor.use_voice_enhance = turn_on
    return {"message": f"Use noise remove: {turn_on}"}

@app.websocket("/ws/room/{room_name}")
async def websocket_endpoint(websocket: WebSocket, room_name: str):
    await audio_processor.join_room(websocket, room_name)

@app.get("/")
async def read_root():
    return {"message": "다중 채팅방 지원 실시간 오디오 처리 서버"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=24015)
