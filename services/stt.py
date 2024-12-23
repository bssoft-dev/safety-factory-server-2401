import aiohttp
from sqlmodel import Session, select
from database import engine
from models import Rooms


class SttProcessor:
    def __init__(self):
        self.infer_url = "https://stt.bs-soft.co.kr/v2401/stt/realtime/byte"
        self.byte_headers = {"Content-Type": "application/octet-stream"}
        self.text_remain_counter = 0

    async def send_audio(self, audio_data: bytes, room_name: str):
        try:
            async with aiohttp.ClientSession() as http_session:
                async with http_session.post(f"{self.infer_url}/{room_name}", data=audio_data, headers=self.byte_headers) as res:
                    if res.status == 200:
                        data = await res.json()
                        if data["result"] != "":
                            print(f"STT: {data}")
                            self.text_remain_counter = min(len(data["result"])//5, 15)
                        else:
                            self.text_remain_counter = self.text_remain_counter - 1
                        if (self.text_remain_counter <= 0) or (data["result"] != ""):
                            with Session(engine) as session:
                                try:
                                    room = session.exec(select(Rooms).filter(Rooms.room_name == room_name)).first()
                                    room.message = data["result"]
                                    session.add(room)
                                    session.commit()
                                except Exception as e:
                                    print(f"STT commit error: {e}")
                    else:
                        raise Exception(f"STT audio failed: {res.status}")
        except Exception as e:
            print(f"STT audio error: {e}")
        