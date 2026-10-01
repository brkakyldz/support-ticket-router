"""Support-ticket router: classify, route by confidence, ask a human when unsure.

    START -> classify --(confidence >= 0.7)--> billing | technical | refund | general -> END
                      `-(confidence <  0.7)--> human_review --(human's category)--^

`langgraph dev` serves the module-level `graph` (no checkpointer: the server persists
threads). Scripts and tests call `build_graph(InMemorySaver())`.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai.chat_models.base import OpenAIRefusalError
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from support_ticket_router.config import get_model, load_env

load_env()

Category = Literal["billing", "technical", "refund", "general"]
BRANCHES: tuple[Category, ...] = ("billing", "technical", "refund", "general")
CONFIDENCE_THRESHOLD = 0.7
STORE = "ShopNest"


class TicketInput(TypedDict):
    message: str


class TicketState(TypedDict, total=False):
    message: str
    category: Category
    confidence: float
    reason: str
    human_note: str
    draft: str
    handled_by: Category


class Classification(BaseModel):
    """How a support ticket should be routed."""

    category: Category
    # No Field constraints or defaults: OpenAI strict JSON schema rejects them. Clamped in code.
    confidence: float = Field(description="0.0 to 1.0: how sure you are that this is the right category")
    reason: str = Field(description="one sentence explaining the choice")


class HumanDecision(BaseModel):
    """The reviewer's answer when the graph pauses: pick a category, add a note, or both."""

    model_config = ConfigDict(extra="forbid")  # a typo such as "categroy" must not pass silently
    category: Category | None = Field(default=None, description="leave empty to keep the model's category")
    note: str = Field(default="", description="guidance for the reply, e.g. what to ask the customer")


CLASSIFY_PROMPT = f"""\
You route customer-support tickets for {STORE}, an online store in Turkey.

Categories:
- billing: payments and charges: duplicate or wrong charges, invoices, payment methods,
  instalments, membership fees, price adjustments.
- technical: the website or app not working: crashes, error pages, login and password
  problems, notifications.
- refund: sending a product back and getting the money for it: return requests, refunds
  for returned, damaged or unwanted items, refund status.
- general: everything else: shipping and delivery, stock, changing an order, store
  policies, opening hours.

Give the single best category, a confidence from 0.0 to 1.0 and a one-sentence reason.
Calibrate the confidence honestly:
- 0.9 or more when the message clearly belongs to one category;
- 0.4 to 0.6 when it could reasonably belong to two categories;
- below 0.4 when the message is too vague to tell what the customer needs.
"""

BRANCH_PROMPTS: dict[Category, str] = {
    "billing": "You are on the billing team. You can explain charges, invoices, payment options "
               "and fees, and open a billing investigation.",
    "technical": "You are on the technical support team. Suggest one or two concrete troubleshooting "
                 "steps and ask for the device, app version or a screenshot if needed.",
    "refund": "You are on the returns and refunds team. Explain the next step of the return or refund "
              "and the usual timeline (refunds reach the card 5-10 working days after the return arrives).",
    "general": "You are on the customer care team. Answer general questions about orders, delivery, "
               "stock and store policies.",
}


# --- model calls (kept apart from the nodes so tests can replace them) --------


def classify_ticket(message: str) -> Classification:
    classifier = get_model().with_structured_output(Classification).with_config(tags=["nostream"])
    return classifier.invoke([SystemMessage(CLASSIFY_PROMPT), HumanMessage(message)])


def draft_reply(branch: Category, message: str, human_note: str = "") -> str:
    system = (
        f"{BRANCH_PROMPTS[branch]}\nWrite a short, friendly reply to the customer (3-5 sentences) "
        f"for {STORE}. Do not promise amounts or dates you cannot know; ask for missing details. "
        f"Sign it '{STORE} {branch.capitalize()} Team'."
    )
    if human_note:
        system += f"\nA human reviewer left this internal note. Follow it, but do not quote it:\n{human_note}"
    return get_model().invoke([SystemMessage(system), HumanMessage(message)]).text


# --- nodes ------------------------------------------------------------------


def classify(state: TicketState) -> dict:
    # A new ticket starts clean: on a reused thread (e.g. in Studio) the previous
    # ticket's reviewer note must not reach this ticket's reply.
    fresh = {"human_note": "", "draft": ""}
    try:
        result = classify_ticket(state["message"])
    except (OpenAIRefusalError, ValidationError, ValueError) as exc:
        # The model refused or returned unparsable output: a human decides. API errors
        # (auth, rate limits, network) are not caught; they should fail loudly.
        return {**fresh, "category": "general", "confidence": 0.0, "reason": f"classifier failed: {exc}"}
    return {
        **fresh,
        "category": result.category,
        "confidence": min(max(result.confidence, 0.0), 1.0),
        "reason": result.reason,
    }


def route_after_classify(state: TicketState) -> Literal["billing", "technical", "refund", "general", "human_review"]:
    """Confident -> straight to the branch; unsure -> a human decides."""
    return state["category"] if state["confidence"] >= CONFIDENCE_THRESHOLD else "human_review"


def human_review(state: TicketState) -> dict:
    # On resume this node runs again from the top, so nothing before interrupt()
    # may have side effects (no model calls here).
    decision: HumanDecision = interrupt(
        {
            "question": "The classifier is not sure. Pick the category (or keep it) and optionally add a note.",
            "message": state["message"],
            "model_category": state["category"],
            "confidence": state["confidence"],
            "reason": state["reason"],
            "options": list(BRANCHES),
        },
        response_schema=HumanDecision,
    )
    return {"category": decision.category or state["category"], "human_note": decision.note}


def route_by_category(state: TicketState) -> Category:
    return state["category"]


def make_branch(name: Category):
    def branch(state: TicketState) -> dict:
        return {"draft": draft_reply(name, state["message"], state.get("human_note", "")), "handled_by": name}

    branch.__name__ = name
    return branch


def build_graph(checkpointer=None):
    builder = StateGraph(TicketState, input_schema=TicketInput)
    builder.add_node("classify", classify)
    builder.add_node("human_review", human_review)
    for name in BRANCHES:
        builder.add_node(name, make_branch(name))
        builder.add_edge(name, END)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route_after_classify)  # targets from the Literal
    builder.add_conditional_edges("human_review", route_by_category, list(BRANCHES))
    return builder.compile(checkpointer=checkpointer, name="ticket_router")


graph = build_graph()
