"""End-to-end acceptance checks against the real model, the dev server and LangSmith.

    uv run python scripts/acceptance.py            # graph shape, sample tickets, LangSmith
    uv run python scripts/acceptance.py --server   # also drive a running `langgraph dev` (port 2024)

Check 4 needs LANGSMITH_API_KEY in .env.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import UTC, datetime, timedelta

from support_ticket_router.config import LANGSMITH_PROJECT, load_env

load_env()

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from support_ticket_router.cli import evaluate, load_samples, run_ticket  # noqa: E402
from support_ticket_router.graph import BRANCHES, build_graph, graph  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, evidence: str) -> None:
    RESULTS.append((name, ok, evidence))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {evidence}\n")


def expected_edges() -> set[tuple[str, str]]:
    edges = {("__start__", "classify"), ("classify", "human_review")}
    for b in BRANCHES:
        edges |= {("classify", b), ("human_review", b), (b, "__end__")}
    return edges


def check_server() -> None:
    from langgraph_sdk import get_sync_client

    client = get_sync_client(url="http://127.0.0.1:2024")
    assistant = client.assistants.search(graph_id="ticket_router")[0]  # Studio may have added more
    drawn = client.assistants.get_graph(assistant["assistant_id"])  # what Studio renders
    edges = {(e["source"], e["target"]) for e in drawn["edges"]}
    nodes = sorted(n["id"] for n in drawn["nodes"])
    check("1b. dev server serves the graph Studio draws", expected_edges() <= edges,
          f"nodes {nodes}; {len(edges)} edges incl. classify -> 5 targets and human_review -> 4 branches")

    ticket = next(t for t in load_samples() if t["id"] == "T16")
    thread = client.threads.create()["thread_id"]
    first = client.runs.wait(thread, "ticket_router", input={"message": ticket["message"]})
    state = client.threads.get_state(thread)
    paused = client.threads.get(thread)["status"] == "interrupted" and state["next"] == ["human_review"]
    payload = state["interrupts"][0]["value"] if state["interrupts"] else {}
    # The resume schema travels with the run output's __interrupt__, not with the thread state.
    schema = (first.get("__interrupt__") or [{}])[0].get("response_schema")
    final = client.runs.wait(thread, "ticket_router", command={"resume": ticket["human_decision"]})
    check("3b. interrupt and resume through the server API (Studio's path)",
          paused and final.get("handled_by") == ticket["human_decision"]["category"],
          f"status interrupted, next {state['next']}, payload keys {sorted(payload)}, "
          f"typed resume form fields {sorted((schema or {}).get('properties', {}))}; "
          f"resumed with {ticket['human_decision']} -> handled_by {final.get('handled_by')!r}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    args = parser.parse_args()

    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    check("1. graph has four branches + human_review", expected_edges() == edges,
          f"{len(edges)} edges: " + ", ".join(f"{s}->{t}" for s, t in sorted(edges)))
    if args.server:
        check_server()

    compiled = build_graph(InMemorySaver())
    results = []
    for ticket in load_samples():
        decision = ticket.get("human_decision", {"category": None, "note": ""})
        result = run_ticket(compiled, ticket["message"], lambda payload, d=decision: d)
        results.append((ticket, result, evaluate(ticket, result)))
        print(f"  {ticket['id']} expected {ticket['expected']:<12} -> "
              f"{'paused, ' if result['review'] else ''}{result['handled_by']} ({result['confidence']:.2f})")
    print()
    clear = [r for r in results if r[0]["expected"] != "human_review"]
    vague = [r for r in results if r[0]["expected"] == "human_review"]
    check("2. clear tickets go to the right branch", all(ok for *_, ok in clear),
          f"{sum(ok for *_, ok in clear)}/{len(clear)} routed directly to the expected branch")
    check("3. ambiguous tickets pause; the human's choice picks the branch", all(ok for *_, ok in vague),
          f"{sum(ok for *_, ok in vague)}/{len(vague)}: " + "; ".join(
              f"{t['id']} conf {r['confidence']:.2f} -> reviewer chose {t['human_decision']['category']} -> "
              f"{r['handled_by']}" for t, r, _ in vague))

    if not os.environ.get("LANGSMITH_API_KEY"):
        check("4. traces in LangSmith show the chosen branch", False,
              "NOT RUN: LANGSMITH_API_KEY is not set")
    else:
        from langchain_core.tracers.langchain import wait_for_all_tracers
        from langsmith import Client

        wait_for_all_tracers()
        client, project = Client(), LANGSMITH_PROJECT  # the project set in config.py, not whatever the env says
        ticket, result, _ = next(r for r in results if r[0]["id"] == "T01")
        names: list[str] = []
        for _ in range(12):  # ingestion is asynchronous
            roots = [r for r in client.list_runs(project_name=project, is_root=True, limit=100,
                                                 start_time=datetime.now(UTC) - timedelta(minutes=30))
                     if ((r.extra or {}).get("metadata") or {}).get("thread_id") == result["thread_id"]]
            if roots:
                names = [r.name for r in client.list_runs(project_name=project, trace_id=roots[0].id)]
                if "route_after_classify" in names and "billing" in names:
                    break
            time.sleep(5)
        check("4. traces in LangSmith show the chosen branch",
              "route_after_classify" in names and "billing" in names,
              f"project {project}, ticket T01 thread {result['thread_id']}: runs in its trace: {sorted(set(names))}")

    passed = sum(ok for _, ok, _ in RESULTS)
    print(f"{passed}/{len(RESULTS)} checks passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
