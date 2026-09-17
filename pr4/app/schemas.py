from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Topic = Literal["доставка", "оплата", "повернення", "гарантія", "підтримка", "інше"]


class AssistantReply(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reply: str = Field(..., description="Текст відповіді для клієнта")
    topic: Topic = Field(..., description="Тема звернення з фіксованого переліку")
    grounded_in_rules: bool = Field(..., description="Чи відповідь ґрунтується на правилах з context.md")
    needs_clarification: bool = Field(..., description="Чи звернення потребує уточнення від клієнта")
    escalate_to_human: bool = Field(..., description="Чи потрібно передати розмову оператору")
    order_number: Optional[str] = Field(None, description="Номер замовлення, якщо клієнт його назвав")


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str = Field(..., min_length=1)


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Ticket(BaseModel):
    topic: Optional[Topic] = None
    order_number: Optional[str] = None
    grounded_in_rules: Optional[bool] = None
    needs_clarification: Optional[bool] = None
    escalate_to_human: bool = False


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    topic: Topic
    grounded_in_rules: bool
    needs_clarification: bool
    escalate_to_human: bool
    order_number: Optional[str]
    ticket: Ticket
    model: str
    latency_ms: float
    usage: Usage
    estimated_tokens: int
    schema_retry_used: bool
