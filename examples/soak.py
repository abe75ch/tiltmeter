"""Soak test: feed real-style support tickets to Jev through Pydantic AI, recorded by Tiltmeter.

Tickets come from the Bitext customer-support dataset (CDLA-Sharing-1.0), cached in soak/
and never committed. One ticket per interval; a Tiltmeter check every hour.

  TYPESAFE_API_KEY=... python examples/soak.py [--interval 60] [--limit N]
"""
import argparse, json, random, sys, time, urllib.request
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic_ai import Agent, UseEnumMemberDocstrings
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import tiltmeter

DATA = ROOT / "soak"
DATASET = "https://datasets-server.huggingface.co/rows?dataset=bitext/Bitext-customer-support-llm-chatbot-training-dataset&config=default&split=train"


class Team(UseEnumMemberDocstrings, str, Enum):
    orders = "orders"
    """Placing, changing or cancelling an order."""
    refunds = "refunds"
    """Refunds, returns and money back."""
    delivery = "delivery"
    """Shipping, delivery dates and tracking."""
    payment = "payment"
    """Payment methods, charges and invoices."""
    account = "account"
    """Creating, accessing or closing an account."""
    other = "other"
    """Anything else, such as feedback or general questions."""


class Triage(BaseModel):
    """Triage a customer support message."""
    model_config = ConfigDict(use_attribute_docstrings=True)
    team: Team
    """Which team should handle this message?"""
    urgent: bool
    """Does the customer need help today?"""
    wants_human: bool
    """Is the customer asking to talk to a person?"""
    mood: Literal["calm", "annoyed", "angry"]
    """How does the customer sound?"""


def tickets(n_pages=20):
    cache = DATA / "tickets.json"
    if cache.exists():
        return json.loads(cache.read_text())
    DATA.mkdir(exist_ok=True)
    rng, out = random.Random(0), []
    for offset in rng.sample(range(0, 26700, 100), n_pages):  # rows are sorted by intent, so sample pages
        with urllib.request.urlopen(f"{DATASET}&offset={offset}&length=100", timeout=30) as r:
            out += [row["row"]["instruction"] for row in json.load(r)["rows"]]
    cache.write_text(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=60)
    ap.add_argument("--limit", type=int, default=0, help="stop after N tickets (0 = run forever)")
    ap.add_argument("--check-every", type=int, default=60, help="tickets between Tiltmeter checks")
    a = ap.parse_args()
    pool, rng, db_path = tickets(), random.Random(), str(DATA / "soak.db")
    provider = TypeSafeProvider(http_client=tiltmeter.instrument(project="soak", db_path=db_path))
    agent = Agent(TypeSafeModel("jev-latest", provider=provider), output_type=Triage)
    print(f"soak: {len(pool)} tickets, one every {a.interval:g}s, check every {a.check_every}", flush=True)
    n = 0
    while not a.limit or n < a.limit:
        text = rng.choice(pool)
        try:
            out = agent.run_sync(text).output
            print(f"{time.strftime('%H:%M:%S')} {out.team.value:<8} urgent={out.urgent!s:<5} human={out.wants_human!s:<5} {out.mood:<7} | {text[:60]}", flush=True)
        except Exception as e:  # keep soaking through transient API errors
            print(f"{time.strftime('%H:%M:%S')} error: {type(e).__name__}: {str(e)[:200]}", flush=True)
        n += 1
        if n % a.check_every == 0:
            con = tiltmeter.db(db_path)
            new, resolved = tiltmeter.track(con, tiltmeter.check(con, {"urgent": 0.5, "wants_human": 0.5}))
            tiltmeter.notify(new, None, resolved)
            print(f"{time.strftime('%H:%M:%S')} check: {len(new)} new, {len(resolved)} resolved alerts", flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
