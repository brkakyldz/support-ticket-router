"""Graph shape and the interrupt/resume flow, with the two model calls replaced."""

from __future__ import annotations

import pydantic
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import support_ticket_router.graph as g


@pytest.fixture
def fake_model(monkeypatch):
    """Replace the classifier and the drafter; returns the list of draft calls."""
    drafts: list[tuple[str, str, str]] = []
    verdicts: dict[str, g.Classification] = {}

    def classify_ticket(message: str) -> g.Classification:
        if message not in verdicts:
            raise ValueError("Structured Output response does not have a 'parsed' field")
        return verdicts[message]

    def draft_reply(branch, message, human_note=""):
        drafts.append((branch, message, human_note))
        return f"[{branch} draft]"

    monkeypatch.setattr(g, "classify_ticket", classify_ticket)
    monkeypatch.setattr(g, "draft_reply", draft_reply)
    return verdicts, drafts


def run(graph, payload, thread="t1"):
    return graph.invoke(payload, {"configurable": {"thread_id": thread}}, version="v2")


def test_graph_has_four_branches_and_human_review():
    edges = {(e.source, e.target, e.conditional) for e in g.graph.get_graph().edges}
    for branch in g.BRANCHES:
        assert ("classify", branch, True) in edges
        assert ("human_review", branch, True) in edges
        assert (branch, "__end__", False) in edges
    assert ("classify", "human_review", True) in edges
    assert ("__start__", "classify", False) in edges


def test_served_graph_has_no_checkpointer():
    # `langgraph dev` refuses to load a module-level graph compiled with one.
    assert g.graph.checkpointer is None


def test_confident_ticket_goes_straight_to_its_branch(fake_model):
    verdicts, drafts = fake_model
    verdicts["charged twice"] = g.Classification(category="billing", confidence=0.95, reason="duplicate charge")
    out = run(g.build_graph(InMemorySaver()), {"message": "charged twice"})
    assert out.interrupts == ()
    assert out.value["handled_by"] == "billing" and out.value["draft"] == "[billing draft]"
    assert drafts == [("billing", "charged twice", "")]


def test_unsure_ticket_pauses_then_follows_the_human(fake_model):
    verdicts, drafts = fake_model
    verdicts["help"] = g.Classification(category="general", confidence=0.4, reason="too vague")
    graph = g.build_graph(InMemorySaver())

    paused = run(graph, {"message": "help"})
    [interrupt] = paused.interrupts
    assert interrupt.value["model_category"] == "general" and interrupt.value["reason"] == "too vague"
    assert interrupt.value["options"] == list(g.BRANCHES)
    assert drafts == []  # nothing drafted while waiting

    done = run(graph, Command(resume={"category": "refund", "note": "ask for the order number"}))
    assert done.interrupts == ()
    assert done.value["handled_by"] == "refund"
    assert drafts == [("refund", "help", "ask for the order number")]


def test_note_only_keeps_the_model_category(fake_model):
    verdicts, _ = fake_model
    verdicts["hmm"] = g.Classification(category="technical", confidence=0.5, reason="maybe the app")
    graph = g.build_graph(InMemorySaver())
    run(graph, {"message": "hmm"})
    done = run(graph, Command(resume={"note": "ask for the app version"}))
    assert done.value["handled_by"] == "technical" and done.value["human_note"] == "ask for the app version"


def test_invalid_resume_is_rejected_and_the_thread_stays_paused(fake_model):
    verdicts, _ = fake_model
    verdicts["hmm"] = g.Classification(category="technical", confidence=0.5, reason="maybe the app")
    graph = g.build_graph(InMemorySaver())
    run(graph, {"message": "hmm"})
    with pytest.raises(pydantic.ValidationError):
        run(graph, Command(resume={"category": "spam"}))
    assert graph.get_state({"configurable": {"thread_id": "t1"}}).next == ("human_review",)
    assert run(graph, Command(resume={"category": "general"})).value["handled_by"] == "general"


def test_classifier_failure_goes_to_a_human(fake_model):
    out = run(g.build_graph(InMemorySaver()), {"message": "unknown to the fake model"})
    assert out.value["confidence"] == 0.0 and out.interrupts


def test_threshold_boundary(fake_model):
    verdicts, _ = fake_model
    verdicts["edge"] = g.Classification(category="refund", confidence=g.CONFIDENCE_THRESHOLD, reason="r")
    verdicts["over"] = g.Classification(category="refund", confidence=1.7, reason="r")  # clamped to 1.0
    graph = g.build_graph(InMemorySaver())
    assert run(graph, {"message": "edge"}, "a").interrupts == ()
    out = run(graph, {"message": "over"}, "b")
    assert out.interrupts == () and out.value["confidence"] == 1.0


def test_api_errors_are_not_mistaken_for_ambiguity(monkeypatch):
    def rate_limited(message):
        raise RuntimeError("429 Too Many Requests")

    monkeypatch.setattr(g, "classify_ticket", rate_limited)
    with pytest.raises(RuntimeError, match="429"):
        run(g.build_graph(InMemorySaver()), {"message": "anything"})


def test_misspelled_resume_keys_are_rejected(fake_model):
    verdicts, _ = fake_model
    verdicts["hmm"] = g.Classification(category="technical", confidence=0.5, reason="maybe the app")
    graph = g.build_graph(InMemorySaver())
    run(graph, {"message": "hmm"})
    with pytest.raises(pydantic.ValidationError):
        run(graph, Command(resume={"categroy": "refund"}))


def test_a_reviewer_note_does_not_leak_into_the_next_ticket_on_the_same_thread(fake_model):
    verdicts, drafts = fake_model
    verdicts["vague"] = g.Classification(category="general", confidence=0.3, reason="vague")
    verdicts["charged twice"] = g.Classification(category="billing", confidence=0.95, reason="clear")
    graph = g.build_graph(InMemorySaver())
    run(graph, {"message": "vague"}, "same")
    run(graph, Command(resume={"note": "Apologise first."}), "same")
    out = run(graph, {"message": "charged twice"}, "same")
    assert out.value["human_note"] == ""
    assert drafts[-1] == ("billing", "charged twice", "")
