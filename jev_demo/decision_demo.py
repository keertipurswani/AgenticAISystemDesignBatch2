import json
import os
import time
from pathlib import Path

from pydantic import BaseModel, Field
from typing_extensions import Literal

from langchain_typesafe import Choice, Noul, Score, TypeSafeClassifier

from dotenv import load_dotenv
import os

load_dotenv()

TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Same ticket for every model. State can be a string or a JSON object.
TICKET = {
    "ticket": (
        "I was charged twice for the October invoice. The second charge "
        "posted this morning. Please refund the duplicate. The app also "
        "showed a 500 while I was paying, but the money left my account twice."
    ),
    "policy": (
        "Duplicate charges are refunded when the customer asks and both "
        "charges are described. A page error alone is a technical issue."
    ),
}

# Three narrow questions. Jev answers them together, independently.
# Choice = one label. Noul = probability of yes. Score = position on a scale.
# https://docs.typesafe.ai/introduction
QUESTIONS = {
    "team": Choice(
        instructions="Which team should handle this ticket?",
        criteria={
            "billing": "Payments, invoices, duplicate charges, and refunds.",
            "technical": "Product bugs, errors, and failed requests.",
            "shipping": "Delivery, tracking, and lost packages.",
            "other": "The ticket does not fit billing, technical, or shipping.",
        },
    ),
    "refund_requested": Noul(
        instructions="Does the customer explicitly ask for money back?",
    ),
    "severity": Score(
        instructions="How severe is the customer impact described in the ticket?",
        criteria=[
            "Inconvenience only. No money missing and the product is usable.",
            "Something is degraded, or a payment looks wrong, but work can continue.",
            "Money is missing, or the customer cannot complete the task at all.",
        ],
    ),
}


class TeamProbabilities(BaseModel):
    """One number per team. A free-form dict is rejected by OpenAI structured output."""

    billing: float
    technical: float
    shipping: float
    other: float


class SeverityProbabilities(BaseModel):
    """One number per severity level, from lowest to highest."""

    level_0: float = Field(description="Probability of severity level 0, inconvenience only.")
    level_1: float = Field(description="Probability of severity level 1, degraded but usable.")
    level_2: float = Field(description="Probability of severity level 2, blocked or money missing.")


class FrontierDecision(BaseModel):
    """Same three judgments, generated as JSON by a chat model.

    OpenAI structured output requires every key to be declared. dict[str, float]
    becomes an open object, and the API rejects that schema.
    https://developers.openai.com/api/docs/guides/structured-outputs
    """

    team: Literal["billing", "technical", "shipping", "other"]
    team_probabilities: TeamProbabilities = Field(
        description="Probability of each team. The four values must sum to 1."
    )
    refund_requested_probability: float = Field(
        description="Probability that the customer explicitly asks for money back, from 0 to 1."
    )
    severity_score: float = Field(description="Expected severity from 0 to 2. May be fractional.")
    severity_probabilities: SeverityProbabilities = Field(
        description="Probability of severity levels 0, 1, and 2. The three values must sum to 1."
    )


def call_jev() -> None:
    """One Jev call. Questions go in invoke(), not on the classifier.

    That is the 0.0.1a3 API. Older alphas stored questions on the constructor.
    """
    print("=== Jev ===")
    classifier = TypeSafeClassifier(
        model="jev-1.13.0",
        api_key=TYPESAFE_API_KEY,
    )

    started = time.perf_counter()
    # state and questions travel together in one request.
    response = classifier.invoke({"state": TICKET, "questions": QUESTIONS})
    elapsed = time.perf_counter() - started

    team = response.choices["team"]
    refund = response.nouls["refund_requested"]
    severity = response.scores["severity"]

    print(f"model:    {response.model}")
    print(f"elapsed:  {elapsed:.3f}s")
    print(f"tokens:   input={response.usage.input_tokens} output={response.usage.output_tokens}")
    print(f"team:     {team.choice}  confidence={team.confidence:.3f}")
    print(f"          probabilities={team.probabilities}")
    # confidence is how peaked the distribution is, not the winning probability.
    print(f"refund:   P(yes)={refund.noul:.3f}")
    # score can fall between the levels you listed. Noul has no confidence field.
    print(f"severity: {severity.score:.3f}  confidence={severity.confidence:.3f}")
    print(f"          probabilities={severity.probabilities}")
    print(f"route:    {_route(team.choice, team.confidence, refund.noul)}")
    print()


def _route(team: str, confidence: float, refund_probability: float) -> str:
    """Your code decides what to do. Jev only returns the judgments.

    0.6 is the example floor from TypeSafe's confidence-routing guide.
    """
    if confidence < 0.6:
        return "human review"
    if team == "billing" and refund_probability >= 0.8:
        return "billing queue, refund review"
    return f"{team} queue"


def call_frontier(model_id: str) -> None:
    """Ask one chat model for the same decision with structured output.

    method='json_schema' is the current native structured-output mode.
    Temperature is left unset because GPT-6 rejects it while reasoning.
    """
    from langchain.chat_models import init_chat_model

    print(f"=== {model_id} ===")
    model = init_chat_model(
        model_id,
        api_key=OPENAI_API_KEY,
        timeout=120,
        max_retries=1,
    )
    structured = model.with_structured_output(
        FrontierDecision,
        method="json_schema",
        include_raw=True,
    )

    started = time.perf_counter()
    result = structured.invoke(
        "Decide this ticket. Return only the schema fields.\n\n" + json.dumps(TICKET, indent=2)
    )
    elapsed = time.perf_counter() - started

    parsed = result["parsed"]
    usage = getattr(result["raw"], "usage_metadata", None) or {}
    print(f"elapsed:  {elapsed:.3f}s")
    print(f"tokens:   input={usage.get('input_tokens')} output={usage.get('output_tokens')}")
    if result["parsing_error"] or parsed is None:
        print(f"parse error: {result['parsing_error']}")
        print()
        return

    print(f"team:     {parsed.team}")
    print(f"          probabilities={parsed.team_probabilities.model_dump()}")
    print(f"refund:   P(yes)={parsed.refund_requested_probability:.3f}")
    print(f"severity: {parsed.severity_score:.3f}")
    print(f"          probabilities={parsed.severity_probabilities.model_dump()}")
    print()


def main() -> None:
    if not TYPESAFE_API_KEY:
        raise SystemExit(f"Set TYPESAFE_API_KEY in {ENV_PATH}")

    call_jev()

    if not OPENAI_API_KEY:
        print(f"OPENAI_API_KEY is empty in {ENV_PATH}, so the frontier comparison was skipped.")
        return

    # FRONTIER_MODELS is a comma-separated list in .env.
    model_ids = "openai:gpt-6-luna,openai:gpt-6-astra"
    for model_id in model_ids.split(","):
        model_id = model_id.strip()
        if not model_id:
            continue
        try:
            call_frontier(model_id)
        except Exception as error:
            print(f"=== {model_id} ===")
            print(f"call failed: {type(error).__name__}: {error}")
            print()


if __name__ == "__main__":
    main()
