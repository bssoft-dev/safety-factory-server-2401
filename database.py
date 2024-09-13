# One line of FastAPI imports here later 👈
from sqlmodel import Field, SQLModel, create_engine
from env import DATABASE_URL

db = 'postgres'
# db_schema = "safety_factory_2401"
# engine = create_engine(f"{DATABASE_URL}/{db}", connect_args={"options": f"-csearch_path={db_schema}"})
engine = create_engine(f"{DATABASE_URL}/{db}")


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)