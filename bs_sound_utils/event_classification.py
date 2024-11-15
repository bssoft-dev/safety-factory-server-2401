import aiohttp
from sqlmodel import Session
from database import engine
from models import Events


class EventClassifier:
    def __init__(self, device):
        self.device = device
        self.infer_url = "https://api-2106.bs-soft.co.kr/v2401/classify-audio/byte"
        self.byte_headers = {"Content-Type": "application/octet-stream"}

    async def classify_audio(self, audio_data: bytes, room_name: str):
        try:
            async with aiohttp.ClientSession() as http_session:
                async with http_session.post(self.infer_url, data=audio_data, headers=self.byte_headers) as res:
                    if res.status == 200:
                        data = await res.json()
                        print(data)
                        if data["is_danger"] == 1:
                            with Session(engine) as session:
                                session.add(Events(room_name=room_name, event=data["event"], time=data["time"]))
                                session.commit()
                    else:
                        raise Exception(f"Classify audio failed: {res.status}")
        except Exception as e:
            print(f"Classify audio error: {e}")
        