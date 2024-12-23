from fastapi import FastAPI, WebSocket, HTTPException
from starlette.middleware.cors import CORSMiddleware
from database import create_db_and_tables, engine, clear_room_data
from sqlmodel import Session, select
from models import Rooms, Events, Devices
from contextlib import asynccontextmanager
from utils.sys import aprint
from routers.log_mon import log_mon
from routers.chat import chat
import datetime


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


app.include_router(log_mon)
app.include_router(chat)

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=24015, reload=True, log_config='utils/log_conf.json')
