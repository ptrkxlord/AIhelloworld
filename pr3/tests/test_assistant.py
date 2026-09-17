from app.assistant import build_messages


def test_messages_are_separated() -> None:
    a = build_messages("  Доставка скільки днів?  ", "  Правила доставки  ")

    assert len(a) == 3
    assert a[0]["role"] == "system"
    assert a[1]["role"] == "system"
    assert a[2] == {"role": "user", "content": "Доставка скільки днів?"}
    assert "Правила магазину:" in a[1]["content"]
    assert "Правила доставки" in a[1]["content"]
