import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import ValidationError

from app import config, prompts, store
from app.llm_client import LLMError, chat
from app.schemas import AssistantReply, ChatRequest, ChatResponse, Ticket, Usage

app = FastAPI(title="PR4 Support Assistant")

ERROR_STATUS = {
    "timeout": 504,
    "rate_limit": 429,
    "server_error": 502,
    "unavailable": 502,
    "auth_error": 401,
    "bad_request": 502,
    "unknown": 502,
}

ORDER_NUMBER_RE = re.compile(r"^SHOP-\d{6}$")

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _strip_fences(text: str) -> str:
    match = FENCE_RE.search(text)
    return match.group(1) if match else text


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return Path(__file__).parent.joinpath("static/index.html").read_text(encoding="utf-8")


def _build_messages(conversation: store.Conversation, user_message: str) -> list[dict]:
    return [
        prompts.build_system_message(),
        prompts.build_context_message(),
        *conversation.budgeted_history(config.HISTORY_TOKEN_BUDGET),
        {"role": "user", "content": user_message},
    ]


def _finalize(parsed: dict) -> dict:
    parsed = dict(parsed)
    if parsed.get("order_number") and not ORDER_NUMBER_RE.match(parsed["order_number"]):
        print(f"[format-check] order_number '{parsed['order_number']}' does not match SHOP-XXXXXX")
    parsed["escalate_to_human"] = parsed["escalate_to_human"] or not parsed["grounded_in_rules"]
    return parsed


def _fallback_reply() -> dict:
    return AssistantReply(
        reply="Вибачте, сталася технічна помилка під час обробки звернення. Я передаю розмову оператору.",
        topic="інше",
        grounded_in_rules=False,
        needs_clarification=False,
        escalate_to_human=True,
        order_number=None,
    ).model_dump()


def _ask_model(messages: list[dict]) -> tuple[dict, dict, bool]:
    schema = AssistantReply.model_json_schema()
    result = chat(messages, response_schema=schema)
    try:
        parsed = AssistantReply.model_validate_json(_strip_fences(result["content"])).model_dump()
        return _finalize(parsed), result, False
    except ValidationError as exc:
        retry_messages = messages + [
            {"role": "assistant", "content": result["content"]},
            {
                "role": "user",
                "content": f"Попередня відповідь не пройшла перевірку JSON Schema: {exc}. Поверни тільки виправлений JSON без пояснень і без огорожі з зворотних лапок.",
            },
        ]
        retry_result = chat(retry_messages, response_schema=schema)
        try:
            parsed = AssistantReply.model_validate_json(_strip_fences(retry_result["content"])).model_dump()
            return _finalize(parsed), retry_result, True
        except ValidationError as retry_exc:
            print(f"[validation-failure] response rejected twice: {retry_exc}")
            return _finalize(_fallback_reply()), retry_result, True


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    conversation_id, conversation = store.get_or_create(payload.conversation_id)
    messages = _build_messages(conversation, payload.message)

    try:
        reply, result, retried = _ask_model(messages)
    except LLMError as exc:
        raise HTTPException(status_code=ERROR_STATUS.get(exc.kind, 502), detail=str(exc))

    conversation.add_turn(payload.message, reply["reply"])
    conversation.update_ticket(reply)

    estimated = sum(store.estimate_tokens(m["content"]) for m in messages)

    return ChatResponse(
        conversation_id=conversation_id,
        reply=reply["reply"],
        topic=reply["topic"],
        grounded_in_rules=reply["grounded_in_rules"],
        needs_clarification=reply["needs_clarification"],
        escalate_to_human=reply["escalate_to_human"],
        order_number=reply["order_number"],
        ticket=Ticket(**conversation.ticket),
        model=result["model"],
        latency_ms=round(result["latency_ms"], 2),
        usage=Usage(**result["usage"]),
        estimated_tokens=estimated,
        schema_retry_used=retried,
    )
