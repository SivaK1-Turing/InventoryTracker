from dataclasses import dataclass
from uuid import UUID
import re
from decimal import Decimal
from typing import ClassVar
import uuid

@dataclass
class Product:
    id: UUID
    name: str
    sku: str
    price: Decimal
    reorder_level: int
    
    # Validation patterns
    _NAME_MIN_LENGTH: ClassVar[int] = 3
    _SKU_PATTERN: ClassVar[re.Pattern] = re.compile(r'^[A-Z0-9]+$')
    
    def __post_init__(self):
        # Validate UUID
        if not isinstance(self.id, UUID):
            try:
                self.id = uuid.UUID(str(self.id))
            except (ValueError, AttributeError):
                raise ValueError("Invalid UUID format")
                
        # Validate name
        if not isinstance(self.name, str):
            raise TypeError("Name must be a string")
        if len(self.name) < self._NAME_MIN_LENGTH:
            raise ValueError(f"Name must be at least {self._NAME_MIN_LENGTH} characters long")
            
        # Validate SKU
        if not isinstance(self.sku, str):
            raise TypeError("SKU must be a string")
        if not self._SKU_PATTERN.match(self.sku):
            raise ValueError("SKU must contain only uppercase letters and numbers")
            
        # Validate price
        if not isinstance(self.price, Decimal):
            try:
                self.price = Decimal(str(self.price))
            except (ValueError, TypeError):
                raise ValueError("Price must be a valid decimal number")
        if self.price <= 0:
            raise ValueError("Price must be greater than zero")
            
        # Validate reorder_level
        if not isinstance(self.reorder_level, int):
            try:
                self.reorder_level = int(self.reorder_level)
            except (ValueError, TypeError):
                raise ValueError("Reorder level must be an integer")