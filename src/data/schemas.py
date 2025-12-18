from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


class QueryUnderstandingOutput(BaseModel):
    query_id: str
    raw_query: str
    locale: str

    intent: str
    rewritten_query: str

    product_type: Optional[str] = None
    brands: List[str] = Field(default_factory=list)
    colors: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    compatibility: List[str] = Field(default_factory=list)

    model_name: str