from pydantic import BaseModel, Field

from app import config


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=config.DEFAULT_TOP_K, ge=1, le=config.MAX_TOP_K)
    min_score: float = Field(default=config.DEFAULT_MIN_SCORE, ge=-1, le=1)
    category: str | None = None
    product: str | None = None
    audience: str | None = None
    status: str | None = None
