from fastapi import FastAPI, WebSocket, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from voice_chat import VoiceChat
from database import create_db_and_tables, engine, clear_room_data
from sqlmodel import Session, select
from models import Rooms
from contextlib import asynccontextmanager

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


class Room(BaseModel):
    room_name: str

voice_chat = VoiceChat()
alert_info = {"warning": False, "message": ""}

@app.get("/v1/rooms")
async def get_info_rooms():
    with Session(engine) as session:
        return session.exec(select(Rooms)).all()
    
@app.post("/v1/create_room")
async def create_room(room: Room):
    res = await voice_chat.create_room(room.room_name)
    with Session(engine) as session:
        item = Rooms(room_name=room.room_name)
        session.add(item)
        session.commit()
    return res

# @app.get("/v1/room/{room_name}/add_person")
# async def add_person_to_room(room_name: str):
#     voice_chat.add_person_to_room(room_name)
#     return {"message": f"Person added to room '{room_name}'"}

# @app.get("/v1/room/{room_name}/remove_person")
# async def remove_person_from_room(room_name: str):
#     voice_chat.remove_person_from_room(room_name)
#     return {"message": f"Person removed from room '{room_name}'"}

@app.delete("/v1/room/{id}")
async def delete_room(id: int):
    res = await voice_chat.delete_room(id)
    if res is None:
        raise HTTPException(status_code=400, detail=f"Room id {id} is not found")
    return res

@app.get("/v1/noise_remove/{turn_on}")
async def use_noise_remove_default_true(turn_on: bool):
    voice_chat.use_voice_enhance = turn_on
    return {"message": f"Use noise remove: {turn_on}"}

@app.get("/v1/hear_me/{turn_on}")
async def hear_my_sound_default_false(turn_on: bool):
    voice_chat.hear_me = turn_on
    return {"message": f"Remove myself voice: {turn_on}"}

@app.get("/v1/keep_test_room/{turn_on}")
async def keep_test_room_default_true(turn_on: bool):
    voice_chat.keep_test_room = turn_on
    return {"message": f"Keep test room: {turn_on}"}

@app.get("/v1/do_stt/{turn_on}")
async def do_stt_default_true(turn_on: bool):
    voice_chat.do_stt = turn_on
    return {"message": f"Do stt: {turn_on}"}

@app.get("/v1/classify_event/{turn_on}")
async def classify_event_default_false(turn_on: bool):
    voice_chat.classify_event = turn_on
    return {"message": f"Classify event: {turn_on}"}
         
@app.websocket("/ws/room/{room_name}/{sr}/{dtype}/{device_id}")
async def websocket_endpoint(room_name: str, sr: int, dtype: str, device_id: str, websocket: WebSocket):
    await voice_chat.join_room(room_name, sr, dtype, device_id, websocket)
 

@app.get("/")
async def read_root():
    return {"message": "다중 채팅방 지원 실시간 오디오 처리 서버"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=24015, reload=True, log_config='utils/log_conf.json')
