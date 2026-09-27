# Tiltmeter

Notices your Jev decisions tilting before anything visibly breaks.

Tiltmeter is a drop-in proxy for TypeSafe's `/v1/systemone` API. Your app calls Tiltmeter
instead of `api.typesafe.ai` and gets Jev's exact answers back. Tiltmeter records every
answer's probabilities and alerts when a question:

| Alert | Meaning |
|-------|---------|
| `version` | the model behind `jev-latest` changed |
| `drift` | the spread of answers moved away from its baseline |
| `edge` | too many answers sit right next to the threshold your code acts on |
| `accuracy` | estimated accuracy fell, computed from the probabilities alone, no labels needed |

Each alert names the likely cause: a model change, an edit to the question, or a change in inputs.

## Try it

One Python file, standard library only.

```sh
python3 tiltmeter.py demo       # simulated Jev switches versions, alerts fire
python3 tiltmeter.py selftest
```

## Use it

```sh
export TYPESAFE_API_KEY=...
python3 tiltmeter.py serve --threshold is_urgent=0.7 --webhook https://hooks.slack.com/...
```

Point your client at `http://127.0.0.1:4790/v1/systemone`, or
`http://127.0.0.1:4790/<project>/v1/systemone` to keep apps apart. Run
`python3 tiltmeter.py check` at any time for a report.

The demo uses simulated answers, not real Jev results.
