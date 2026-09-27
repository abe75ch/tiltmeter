"""Support triage with Pydantic AI and Jev, recorded by Tiltmeter in-process.

  pip install "pydantic-ai-slim[typesafe]"
  TYPESAFE_API_KEY=... python examples/pydantic_ai_triage.py
  python tiltmeter.py check --db triage.db
"""
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent, BoolCriteria, UseEnumMemberDocstrings
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tiltmeter


class Area(UseEnumMemberDocstrings, str, Enum):
    billing = "billing"
    """Charges, invoices, plans and payment methods."""
    bug = "bug"
    """Part of the product does not work as it should."""
    account = "account"
    """Logging in, access, and account settings."""


class Ticket(BaseModel):
    """Triage a support ticket."""
    model_config = ConfigDict(use_attribute_docstrings=True)
    area: Area
    """Which team owns this ticket?"""
    urgent: Annotated[bool, BoolCriteria(true="Customer losing money or facing a deadline today.", false="Can wait.")]
    """Should this ticket jump the queue?"""
    app: Literal["web", "ios", "android"] | None
    """Which app is it about?"""


TICKETS = [
    "Timeline blank on Android since update, standup in 10 mins.",
    "I was charged twice for September, please refund the duplicate.",
    "Can't log in on the website, password reset email never arrives.",
    "The iOS app crashes every time I open settings.",
    "How do I add a teammate to my workspace?",
]

provider = TypeSafeProvider(http_client=tiltmeter.instrument(project="triage", db_path="triage.db"))
agent = Agent(TypeSafeModel("jev-latest", provider=provider), output_type=Ticket)

for text in TICKETS:
    result = agent.run_sync(text)
    conf = (result.response.provider_details or {}).get("confidence", {})
    print(f"{result.output!s:<55} confidence={conf} | {text[:45]}")
