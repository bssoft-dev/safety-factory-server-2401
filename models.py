from typing import Optional
from pydantic import BaseModel
from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel

db_schema = "safety_factory_2401"

class Room(BaseModel):
    room_name: str

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
    room_name: str
    event: str
    time: str

class Devices(SQLModel, table=True):
    __table_args__ = {'schema': db_schema}
    # __table_args__ = (UniqueConstraint("roomId"),)
    id: Optional[int] = Field(default=None, primary_key=True) # 자동 넘버링
    device_id: str
    owner: str
    note: str =''