from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class StockTransaction(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    product_id: UUID
    delta: int
    timestamp: datetime = Field(default_factory=datetime.now)
    note: Optional[str] = None
    
    @validator('delta')
    def delta_must_not_be_zero(cls, v):
        if v == 0:
            raise ValueError('delta cannot be zero')
        return v
    
    @validator('timestamp')
    def timestamp_not_in_future(cls, v):
        if v > datetime.now():
            raise ValueError('timestamp cannot be in the future')
        return v