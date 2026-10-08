from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    product: str | None = Field(default=None, max_length=80)


class RAGAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=4000)
    source_ids: list[int] = Field(default_factory=list, max_length=8)
    found: bool
