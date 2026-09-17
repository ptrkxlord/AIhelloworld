import uuid
from typing import Optional

from app import config


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / config.CHARS_PER_TOKEN))


class Conversation:
    def __init__(self) -> None:
        self.history: list[dict] = []
        self.ticket: dict = {
            "topic": None,
            "order_number": None,
            "grounded_in_rules": None,
            "needs_clarification": None,
            "escalate_to_human": False,
        }

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": assistant_text})

    def update_ticket(self, reply: dict) -> None:
        self.ticket["topic"] = reply["topic"]
        self.ticket["grounded_in_rules"] = reply["grounded_in_rules"]
        self.ticket["needs_clarification"] = reply["needs_clarification"]
        if reply.get("order_number"):
            self.ticket["order_number"] = reply["order_number"]
        if reply["escalate_to_human"]:
            self.ticket["escalate_to_human"] = True

    def budgeted_history(self, budget: int) -> list[dict]:
        if not self.history:
            return []
        first = self.history[0]
        rest = self.history[1:]
        kept_tail: list[dict] = []
        used = estimate_tokens(first["content"])
        for message in reversed(rest):
            cost = estimate_tokens(message["content"])
            if used + cost > budget:
                break
            kept_tail.insert(0, message)
            used += cost
        return [first] + kept_tail


_conversations: dict[str, Conversation] = {}


def get_or_create(conversation_id: Optional[str]) -> tuple[str, Conversation]:
    if conversation_id and conversation_id in _conversations:
        return conversation_id, _conversations[conversation_id]
    new_id = conversation_id or str(uuid.uuid4())
    conversation = Conversation()
    _conversations[new_id] = conversation
    return new_id, conversation
