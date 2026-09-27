# Tiltmeter

Notices your Jev decisions tilting before anything visibly breaks.

![Tiltmeter demo: a simulated Jev version switch fires alerts](docs/demo.svg)

Jev, TypeSafe's System One model, answers with probabilities instead of text. Tiltmeter
records those probabilities for every question your app asks and alerts when one of them
moves:

| Alert | Meaning |
|-------|---------|
| `version` | the model behind `jev-latest` changed |
| `drift` | the spread of answers moved away from its baseline |
| `edge` | too many answers sit right next to the threshold your code acts on |
| `accuracy` | estimated accuracy fell, computed from the probabilities alone, no labels needed |

Each alert names the likely cause: a model change, an edit to the question, or a change in
your inputs. It fires once, stays quiet while it persists, and is reported again when it
resolves.

Tiltmeter stores probabilities, the model name and a fingerprint of each question. It never
stores the text you send to Jev.

## Install

```sh
pip install "git+https://github.com/abe75ch/tiltmeter"                 # core, standard library only
pip install "tiltmeter[pydantic-ai] @ git+https://github.com/abe75ch/tiltmeter"   # with the Pydantic AI wrapper
tiltmeter demo        # a simulated Jev switches versions and the alerts fire
tiltmeter selftest
```

## Use it with Pydantic AI, no proxy

Give Pydantic AI's TypeSafe provider a Tiltmeter client. Your agent gets the same answers,
and a recording problem is logged, never raised into your app.

```python
from pydantic_ai import Agent
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider
import tiltmeter

provider = TypeSafeProvider(http_client=tiltmeter.instrument(project="support"))
agent = Agent(TypeSafeModel("jev-latest", provider=provider), output_type=Ticket)
```

Questions are named after your output model's fields. See `examples/pydantic_ai_triage.py`.

## Use it as a proxy, any client

```sh
tiltmeter serve --threshold urgent=0.7 --webhook https://hooks.slack.com/services/...
export TYPESAFE_BASE_URL=http://127.0.0.1:4790     # read by the TypeSafe SDK and Pydantic AI
```

Or point any HTTP client at `http://127.0.0.1:4790/v1/systemone`, or at
`http://127.0.0.1:4790/<project>/v1/systemone` to keep apps apart. Callers send their own
API key as usual: the proxy never adds one, and it accepts only JSON requests, so other
programs and web pages can't spend your quota through it. Status codes, headers and
bodies pass through unchanged, and an unreachable TypeSafe returns a clear 502 or 504.

Run `tiltmeter check` at any time for a report. Set thresholds per question
(`--threshold urgent=0.7`) or per project and question (`--threshold support/urgent=0.7`).

## Limitations

- The accuracy estimate assumes Jev's probabilities stay calibrated. It is an early
  warning, not a replacement for checking labelled examples.
- A question needs at least 50 answers in each of two windows of up to 200 before it is
  checked. A drift or accuracy alert needs a real effect size and statistical significance
  (p < 0.001), so small samples and questions with many options don't raise false alarms:
  in simulation, unchanging traffic raised any alert in at most 0.3% of checks, while a
  moderate real shift was caught every time.
- Each window is compared with the one just before it, so a very slow, steady slide can
  stay under the limits. Compare against an older export if you suspect one.
- Estimated accuracy uses the chosen option's probability, not TypeSafe's `confidence`,
  which measures how concentrated the answers are. Score questions get no accuracy
  estimate, since a score is not a right-or-wrong decision.
- The in-process wrapper covers the async client that Pydantic AI and `AsyncTypeSafeClient`
  use. For the synchronous `TypeSafeClient`, use the proxy.
- Data lives in one local SQLite file. It suits a single app or machine, not a fleet.
- The demo and self-check use simulated answers, not real Jev results.

## Not affiliated

Tiltmeter is an independent project. It is not made, endorsed or supported by TypeSafe AI.
Jev and TypeSafe are their owners' names.

## License

MIT
