from __future__ import annotations

from pydantic import BaseModel, field_validator

MAX_QUERY_LENGTH = 200
ALLOWED_LOCALES = {"us", "es", "jp"}


class SearchRequest(BaseModel):
    query: str
    locale: str = "us"

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be empty")
        if len(v) > MAX_QUERY_LENGTH:
            raise ValueError(f"query must be at most {MAX_QUERY_LENGTH} characters")
        return v

    @field_validator("locale")
    @classmethod
    def validate_locale(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ALLOWED_LOCALES:
            raise ValueError(f"locale must be one of: {sorted(ALLOWED_LOCALES)}")
        return v
