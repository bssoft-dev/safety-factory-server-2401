# One line of FastAPI imports here later 👈
from sqlmodel import Field, SQLModel, create_engine
from env import DATABASE_URL

db = 'test_cho'
engine = create_engine(f"{DATABASE_URL}/{db}")


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)