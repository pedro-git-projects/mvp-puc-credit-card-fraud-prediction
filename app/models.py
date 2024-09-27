from pydantic import BaseModel


class Transaction(BaseModel):
    time_elapsed: int
    amt: float
    lat: float
    long: float
