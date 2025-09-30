from typing import Optional, Union
from pydantic import BaseModel
from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel
from datetime import datetime

db_schema = "safety_factory_2401"

class Room(BaseModel):
    room_name: str

class RoomSettingsCreate(BaseModel):
    """방 설정 생성/수정용 모델"""
    room_name: str
    use_voice_enhance: bool = True      # 소음제거
    hear_me: bool = False               # 자신의 목소리 듣기
    record_audio: bool = True           # 오디오 녹음
    classify_event: bool = True         # 음원분류 (비명, 경보음 등)
    do_stt: bool = True                 # STT (음성인식)
    enhance_volume: int = 0             # 볼륨 증폭

class RoomSettings(SQLModel, table=True):
    """방별 오디오 처리 설정"""
    __table_args__ = {'schema': db_schema}
    id: Optional[int] = Field(default=None, primary_key=True)
    room_name: str = Field(unique=True, index=True)
    use_voice_enhance: bool = Field(default=True, description="소음제거")
    hear_me: bool = Field(default=False, description="자신의 목소리 듣기")
    record_audio: bool = Field(default=True, description="오디오 녹음")
    classify_event: bool = Field(default=True, description="음원분류")
    do_stt: bool = Field(default=True, description="STT")
    enhance_volume: int = Field(default=0, description="볼륨 증폭")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class Rooms(SQLModel, table=True):
    __table_args__ = {'schema': db_schema}
    # __table_args__ = (UniqueConstraint("roomId"),)
    id: Optional[int] = Field(default=None, primary_key=True) # 자동 넘버링
    room_name: str
    persons: str =''
    message: str =''
    num_person: Optional[int] = 0

class Events(SQLModel, table=True):
    __table_args__ = {'schema': db_schema}
    # __table_args__ = (UniqueConstraint("roomId"),)
    id: Optional[int] = Field(default=None, primary_key=True) # 자동 넘버링
    room_name: str = ''
    device_id: Optional[str] = None
    event: str
    time: str

class Devices(SQLModel, table=True):
    __table_args__ = {'schema': db_schema}
    # __table_args__ = (UniqueConstraint("roomId"),)
    id: Optional[int] = Field(default=None, primary_key=True) # 자동 넘버링
    device_id: str
    owner: str
    note: str =''

class SttRecords(SQLModel, table=True):
    __table_args__ = {'schema': db_schema}
    id: Optional[int] = Field(default=None, primary_key=True)
    room_name: str
    worker: str  # 발화자
    stt_text: str  # STT 결과 텍스트
    wav_file_path: str  # 원음 파일 경로
    created_at: datetime
