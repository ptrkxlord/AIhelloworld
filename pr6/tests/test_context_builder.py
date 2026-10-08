from app.context_builder import build_context


def result(source, audience="клієнти", status="чинний", product="усі товари", text="Правило 12 місяців."):
    return {
        "source": source,
        "section": "Умови",
        "text": text,
        "score": 0.9,
        "metadata": {
            "title": source,
            "audience": audience,
            "status": status,
            "product": product,
            "revision_date": "2026-01-01",
        },
    }


def test_internal_documents_never_enter_context():
    sources = build_context([result("internal.md", audience="персонал")], "правила оператора")
    assert sources == []


def test_archived_documents_need_explicit_historical_question():
    archive = result("old.md", status="архівний")
    assert build_context([archive], "Скільки коштує доставка зараз?") == []
    assert build_context([archive], "Якою була доставка у 2025 році?")[0]["source"] == "old.md"
    assert build_context([archive], "Покажи стару редакцію правил")[0]["source"] == "old.md"


def test_product_filter_keeps_general_and_selected_product_documents():
    sources = build_context([
        result("general.md"),
        result("watch.md", product="Orbit S"),
        result("speaker.md", product="SoundBox Mini"),
    ], "інструкція", product="Orbit S")
    assert {item["source"] for item in sources} == {"general.md", "watch.md"}


def test_context_has_consecutive_reference_ids():
    sources = build_context([result("one.md"), result("two.md")], "питання")
    assert [item["id"] for item in sources] == [1, 2]
