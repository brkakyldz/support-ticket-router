"""Command-line entry points.

    uv run ticket-route "My card was charged twice"   # one ticket; you are the reviewer if it pauses
    uv run ticket-samples                            # all sample tickets -> expected / found table
    uv run ticket-samples --only T16 T18 --drafts    # a subset, printing the drafted replies
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from support_ticket_router.graph import BRANCHES, CONFIDENCE_THRESHOLD, build_graph

SAMPLES = Path(__file__).resolve().parents[2] / "samples" / "tickets.json"
BOLD, DIM, GREEN, RED, YELLOW, CYAN, RESET = (
    "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[0m",
)


def run_ticket(graph, message: str, decide) -> dict:
    """Route one ticket on a fresh thread. `decide(payload)` answers a human_review pause."""
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    out = graph.invoke({"message": message}, config, version="v2")
    review = None
    if out.interrupts:
        review = out.interrupts[0].value
        out = graph.invoke(Command(resume=decide(review)), config, version="v2")
    return {**out.value, "review": review, "thread_id": config["configurable"]["thread_id"]}


def ask_reviewer(payload: dict) -> dict:
    print(f"\n{YELLOW}{BOLD}── human review needed ─────────────────────────{RESET}")
    print(f"model says {BOLD}{payload['model_category']}{RESET} at confidence {payload['confidence']:.2f}")
    print(f"reason: {payload['reason']}")
    options = "/".join(BRANCHES)
    while True:
        category = input(f"{YELLOW}category{RESET} [{options}, Enter = keep]: ").strip().lower()
        if category in ("", *BRANCHES):
            break
    note = input(f"{YELLOW}note for the reply{RESET} (optional): ").strip()
    print(f"{YELLOW}{BOLD}────────────────────────────────────────────────{RESET}\n")
    return {"category": category or None, "note": note}


def route() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Route one support ticket.")
    parser.add_argument("message", nargs="*", help="ticket text (prompted for if omitted)")
    args = parser.parse_args()
    message = " ".join(args.message) or input(f"{CYAN}ticket{RESET}: ")
    result = run_ticket(build_graph(InMemorySaver()), message, ask_reviewer)
    print(f"{DIM}classified {result['review']['model_category'] if result['review'] else result['category']} "
          f"at {result['confidence']:.2f}: {result['reason']}{RESET}")
    reviewed = " after human review" if result["review"] else ""
    print(f"{GREEN}{BOLD}→ {result['handled_by']}{RESET}{reviewed}\n\n{result['draft']}")


def load_samples(only: list[str] | None = None) -> list[dict]:
    tickets = json.loads(SAMPLES.read_text(encoding="utf-8"))
    if not only:
        return tickets
    wanted = {i.upper() for i in only}
    unknown = wanted - {t["id"] for t in tickets}
    if unknown:
        raise SystemExit(f"unknown ticket ids: {', '.join(sorted(unknown))}")
    return [t for t in tickets if t["id"] in wanted]


def evaluate(ticket: dict, result: dict) -> bool:
    if result["reason"].startswith("classifier failed"):
        return False  # a pause caused by a failure is not a correct pause
    if ticket["expected"] == "human_review":
        return result["review"] is not None and result["handled_by"] == ticket["human_decision"]["category"]
    return result["review"] is None and result["handled_by"] == ticket["expected"]


def samples() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run the sample tickets and compare with the expected route.")
    parser.add_argument("--only", nargs="*", help="ticket ids, e.g. T16 T18")
    parser.add_argument("--drafts", action="store_true", help="print each drafted reply")
    args = parser.parse_args()

    graph = build_graph(InMemorySaver())
    rows = []
    print(f"threshold {CONFIDENCE_THRESHOLD}; sending scripted reviewer decisions for pauses\n")
    header = f"{'id':<4} {'expected':<13} {'classified':<10} {'conf':>4}  {'paused':<6} {'handled by':<10} ok"
    print(header + "\n" + "-" * len(header))
    for ticket in load_samples(args.only):
        decision = ticket.get("human_decision", {"category": None, "note": ""})
        result = run_ticket(graph, ticket["message"], lambda payload, d=decision: d)
        classified = result["review"]["model_category"] if result["review"] else result["category"]
        ok = evaluate(ticket, result)
        rows.append((ticket, result, ok))
        mark = f"{GREEN}yes{RESET}" if ok else f"{RED}NO{RESET}"
        print(f"{ticket['id']:<4} {ticket['expected']:<13} {classified:<10} {result['confidence']:>4.2f}  "
              f"{'yes' if result['review'] else '-':<6} {result['handled_by']:<10} {mark}")
        if args.drafts:
            print(f"{DIM}{result['draft']}{RESET}\n")

    clear = [r for r in rows if r[0]["expected"] != "human_review"]
    vague = [r for r in rows if r[0]["expected"] == "human_review"]
    print(f"\nclear tickets routed directly to the expected branch: {sum(ok for *_, ok in clear)}/{len(clear)}")
    print(f"ambiguous tickets paused and resumed into the reviewer's branch: {sum(ok for *_, ok in vague)}/{len(vague)}")
    return 0 if all(ok for *_, ok in rows) else 1
