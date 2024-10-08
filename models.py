from typing import Optional, List
from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel, MetaData

db_schema = "safety_factory_2401"

class Rooms(SQLModel, table=True):
    __table_args__ = {'schema': db_schema}
    # __table_args__ = (UniqueConstraint("roomId"),)
    id: Optional[int] = Field(default=None, primary_key=True) # 자동 넘버링
    room_name: str
    num_person: Optional[int] = 0
