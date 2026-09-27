#!/usr/bin/env python3
"""Tiltmeter: notices your Jev decisions tilting before anything visibly breaks.

A drop-in proxy for TypeSafe's /v1/systemone API. Your app calls Tiltmeter instead
of api.typesafe.ai and gets Jev's exact answers back. Tiltmeter records every
answer's probabilities and checks each question for:

  version   the model behind jev-latest changed (the response names it)
  drift     the spread of answers moved away from its baseline (PSI)
  edge      too many answers sit right next to your decision threshold
  accuracy  estimated accuracy fell, computed from the probabilities alone,
            no labels needed (valid while Jev's probabilities stay calibrated)

  tiltmeter.py serve [--port 4790] [--threshold is_urgent=0.7] [--webhook URL]
  tiltmeter.py check [--threshold is_urgent=0.7] [--webhook URL]
  tiltmeter.py demo          # simulated Jev, a model switch, alerts firing
  tiltmeter.py selftest

Point your client at http://127.0.0.1:4790/v1/systemone, or
http://127.0.0.1:4790/<project>/v1/systemone to keep apps apart. The TypeSafe SDK and
Pydantic AI both read TYPESAFE_BASE_URL, so setting it to http://127.0.0.1:4790 is enough.

Or skip the proxy and record in-process with Pydantic AI (needs pydantic-ai-slim[typesafe]):

  from pydantic_ai.providers.typesafe import TypeSafeProvider
  import tiltmeter
  provider = TypeSafeProvider(http_client=tiltmeter.instrument(project="support"))
"""
import argparse, hashlib, json, math, os, random, sqlite3, sys, threading, time, urllib.error, urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("TYPESAFE_ENDPOINT", "https://api.typesafe.ai")
DB = os.environ.get("TILTMETER_DB", "tiltmeter.db")
WINDOW = 200        # answers per window: recent vs the baseline before it
MIN_N = 50          # skip a question until each window has this many answers
PSI_ALERT = 0.2     # common rule of thumb: >0.1 moderate shift, >0.25 major
EDGE_BAND = 0.05    # "near the threshold" means within this distance
EDGE_ALERT = 0.2    # alert when this share of recent answers is near the threshold
ACC_DROP = 0.05     # alert when estimated accuracy falls by this much


def db(path=None):
    con = sqlite3.connect(path or DB)
    con.execute("""create table if not exists answers(id integer primary key, ts real, project text, model text,
                   qid text, qhash text, qtype text, value real, choice text, conf real, latency_ms real)""")
    return con


def record(con, project, req, resp, latency_ms, ts=None):
    """Store one row per answered question."""
    questions, model, ts = req.get("questions", {}), resp.get("model", "?"), ts or time.time()
    rows = []
    for qid, a in resp.get("answers", {}).items():
        spec = questions.get(qid, {})
        qhash = hashlib.sha1(json.dumps([spec.get("instructions"), spec.get("criteria")], sort_keys=True).encode()).hexdigest()[:10]
        t = a.get("type")
        if t == "noul":
            rows.append((ts, project, model, qid, qhash, t, a["noul"], None, None, latency_ms))
        elif t == "choice":
            rows.append((ts, project, model, qid, qhash, t, a["probabilities"][a["choice"]], a["choice"], a.get("confidence"), latency_ms))
        elif t == "score":
            rows.append((ts, project, model, qid, qhash, t, a["score"], None, a.get("confidence"), latency_ms))
    con.executemany("insert into answers(ts,project,model,qid,qhash,qtype,value,choice,conf,latency_ms) values (?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()


# ---------------------------------------------------------------- checks

def psi(base, recent):
    keys = set(base) | set(recent)
    nb, nr = sum(base.values()), sum(recent.values())
    total = 0.0
    for k in keys:  # add half a count per bucket so an empty bucket doesn't inflate small samples
        b = (base.get(k, 0) + 0.5) / (nb + 0.5 * len(keys))
        r = (recent.get(k, 0) + 0.5) / (nr + 0.5 * len(keys))
        total += (r - b) * math.log(r / b)
    return total


def buckets(rows, qtype):
    if qtype == "noul":
        return Counter(min(int(v * 10), 9) for v, _, _ in rows)
    if qtype == "choice":
        return Counter(c for _, c, _ in rows)
    return Counter(round(v) for v, _, _ in rows)


def est_accuracy(rows, qtype, t):
    """Expected share of correct decisions if the probabilities are calibrated."""
    if qtype == "noul":
        return sum(v if v >= t else 1 - v for v, _, _ in rows) / len(rows)
    return sum((conf if conf is not None else v) for v, _, conf in rows) / len(rows)


def majority(xs):
    return Counter(xs).most_common(1)[0][0]


def check(con, thresholds=None, window=WINDOW, min_n=MIN_N):
    thresholds, alerts = thresholds or {}, []
    groups = con.execute("select distinct project, qid, qtype from answers").fetchall()
    for project, qid, qtype in groups:
        rows = con.execute("select model, qhash, value, choice, conf from answers where project=? and qid=? order by ts, id",
                           (project, qid)).fetchall()
        recent, base = rows[-window:], rows[:-window][-window:]
        if len(base) < min_n or len(recent) < min_n:
            continue
        name = f"{project}/{qid}"
        bm, rm = majority(r[0] for r in base), majority(r[0] for r in recent)
        edited = majority(r[1] for r in base) != majority(r[1] for r in recent)
        cause = "question text changed" if edited else (f"model {bm} -> {rm}" if bm != rm else "inputs or model behaviour changed")
        if bm != rm:
            alerts.append(("version", name, f"model changed {bm} -> {rm}"))
        b, r = [x[2:] for x in base], [x[2:] for x in recent]
        shift = psi(buckets(b, qtype), buckets(r, qtype))
        if shift > PSI_ALERT:
            alerts.append(("drift", name, f"answer distribution shifted, PSI {shift:.2f}; likely cause: {cause}"))
        t = thresholds.get(qid, 0.5)
        if qtype in ("noul", "choice") and qid in thresholds:
            vals = [v if qtype == "noul" else (conf if conf is not None else v) for v, _, conf in r]
            edge = sum(abs(v - t) < EDGE_BAND for v in vals) / len(vals)
            if edge > EDGE_ALERT:
                alerts.append(("edge", name, f"{edge:.0%} of recent answers within {EDGE_BAND} of threshold {t}: decisions are fragile"))
        ab, ar = est_accuracy(b, qtype, t), est_accuracy(r, qtype, t)
        if ab - ar > ACC_DROP:
            alerts.append(("accuracy", name, f"estimated accuracy {ab:.0%} -> {ar:.0%} (no labels; assumes calibration); likely cause: {cause}"))
    return alerts


def track(con, alerts):
    """An alert fires once, stays quiet while it persists, and is reported resolved when it clears."""
    con.execute("""create table if not exists alerts(kind text, name text, msg text, first_ts real, last_ts real,
                   active integer, primary key(kind, name))""")
    now, current = time.time(), {(k, n): m for k, n, m in alerts}
    active = {tuple(r) for r in con.execute("select kind, name from alerts where active=1")}
    new = [(k, n, m) for (k, n), m in current.items() if (k, n) not in active]
    resolved = [(k, n) for k, n in active if (k, n) not in current]
    for k, n, m in new:
        con.execute("""insert into alerts values (?,?,?,?,?,1) on conflict(kind, name)
                       do update set msg=excluded.msg, first_ts=excluded.first_ts, last_ts=excluded.last_ts, active=1""",
                    (k, n, m, now, now))
    for (k, n), m in current.items():
        if (k, n) in active:
            con.execute("update alerts set msg=?, last_ts=? where kind=? and name=?", (m, now, k, n))
    for k, n in resolved:
        con.execute("update alerts set active=0, last_ts=? where kind=? and name=?", (now, k, n))
    con.commit()
    return new, resolved


def notify(alerts, webhook=None, resolved=()):
    for kind, name, msg in alerts:
        print(f"ALERT [{kind}] {name}: {msg}", flush=True)
    for kind, name in resolved:
        print(f"RESOLVED [{kind}] {name}", flush=True)
    if webhook and (alerts or resolved):
        lines = [f"[{k}] {n}: {m}" for k, n, m in alerts] + [f"resolved [{k}] {n}" for k, n in resolved]
        body = json.dumps({"text": "\n".join(lines)}).encode()
        try:  # Slack-compatible incoming webhook
            urllib.request.urlopen(urllib.request.Request(webhook, body, {"Content-Type": "application/json"}), timeout=10)
        except (urllib.error.URLError, OSError) as e:
            print(f"webhook failed: {e}", file=sys.stderr)


# ---------------------------------------------------------------- in-process wrapper

def instrument(project="default", db_path=None, transport=None, **client_kwargs):
    """An httpx2.AsyncClient that records every Jev answer, for TypeSafeProvider(http_client=...)
    or AsyncTypeSafeClient. Your app gets the exact same responses."""
    import httpx2  # installed with pydantic-ai-slim[typesafe] and typesafe-sdk
    path = db_path or DB

    class Recorder(httpx2.AsyncBaseTransport):
        def __init__(self):
            self.inner = transport or httpx2.AsyncHTTPTransport()

        async def handle_async_request(self, request):
            start = time.time()
            response = await self.inner.handle_async_request(request)
            if request.url.path.endswith("/v1/systemone") and response.status_code == 200:
                body = await response.aread()  # cached, so the caller still reads it normally
                try:
                    con = db(path)  # ponytail: a blocking sqlite write of a few ms; move to a queue if it ever shows up in latency
                    record(con, project, json.loads(request.content), json.loads(body), (time.time() - start) * 1000, ts=start)
                    con.close()
                except (ValueError, KeyError, sqlite3.Error) as e:
                    print(f"tiltmeter record failed: {e}", file=sys.stderr)
            return response

        async def aclose(self):
            await self.inner.aclose()

    return httpx2.AsyncClient(transport=Recorder(), **client_kwargs)


# ---------------------------------------------------------------- proxy

def make_handler(upstream, db_path):
    class Proxy(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            parts = [p for p in self.path.split("/") if p]
            if parts[-2:] != ["v1", "systemone"]:
                self.send_error(404, "use /v1/systemone or /<project>/v1/systemone")
                return
            project = "/".join(parts[:-2]) or "default"
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            auth = self.headers.get("Authorization") or f"Bearer {os.environ.get('TYPESAFE_API_KEY', '')}"
            req = urllib.request.Request(f"{upstream}/v1/systemone", body, {"Authorization": auth, "Content-Type": "application/json"})
            start = time.time()
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    status, data = r.status, r.read()
            except urllib.error.HTTPError as e:
                status, data = e.code, e.read()
            latency = (time.time() - start) * 1000
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)  # the caller gets Jev's bytes unchanged
            if status == 200:
                try:
                    con = db(db_path)  # ponytail: one connection per request; pool it if write volume ever matters
                    record(con, project, json.loads(body), json.loads(data), latency, ts=start)  # request time, not write time
                    con.close()
                except (ValueError, KeyError, sqlite3.Error) as e:
                    print(f"record failed: {e}", file=sys.stderr)
    return Proxy


def serve(port, upstream, db_path, thresholds, webhook, every=300):
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(upstream, db_path))
    print(f"tiltmeter proxy on http://127.0.0.1:{port}/v1/systemone -> {upstream}, checks every {every}s", flush=True)

    def loop():
        while True:
            time.sleep(every)
            con = db(db_path)
            new, resolved = track(con, check(con, thresholds))
            con.close()
            notify(new, webhook, resolved)
    threading.Thread(target=loop, daemon=True).start()
    srv.serve_forever()


# ---------------------------------------------------------------- demo: a simulated Jev

def fake_jev(state):
    """Stand-in upstream whose behaviour changes when state['model'] flips."""
    class Fake(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            rng, new = state["rng"], state["model"] != "jev-1.12.0"
            p = min(max(rng.gauss(0.62 if new else 0.2, 0.12 if new else 0.15), 0.0), 1.0)
            dist = {"billing": 0.6, "technical": 0.3, "sales": 0.1} if not new else {"billing": 0.3, "technical": 0.55, "sales": 0.15}
            pick = rng.choices(list(dist), weights=list(dist.values()))[0]
            top = rng.uniform(0.7, 0.97) if not new else rng.uniform(0.45, 0.8)
            probs = {k: (top if k == pick else (1 - top) / 2) for k in dist}
            ans = {"model": state["model"], "answers": {
                "is_urgent": {"type": "noul", "noul": p},
                "department": {"type": "choice", "choice": pick, "probabilities": probs, "confidence": top}},
                "usage": {"input_tokens": 300, "output_tokens": 20}}
            data = json.dumps(ans).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    return Fake


def demo():
    import tempfile
    path = os.path.join(tempfile.mkdtemp(prefix="tiltmeter-"), "demo.db")
    state = {"model": "jev-1.12.0", "rng": random.Random(7)}
    up = ThreadingHTTPServer(("127.0.0.1", 0), fake_jev(state))
    px = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(f"http://127.0.0.1:{up.server_port}", path))
    for s in (up, px):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{px.server_port}/support-app/v1/systemone"
    body = json.dumps({"model": "jev-latest", "state": "ticket text", "questions": {
        "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"billing": None, "technical": None, "sales": None}}}}).encode()

    def send(n):
        for _ in range(n):
            urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"})).read()
        sent[0] += n
        while db(path).execute("select count(*) from answers").fetchone()[0] < sent[0] * 2:
            time.sleep(0.05)  # the proxy saves answers after replying; wait for the last ones

    sent = [0]

    thresholds = {"is_urgent": 0.7, "department": 0.7}
    print("Simulated Jev, not real Jev numbers.")
    print("1. 300 support tickets through the proxy on jev-1.12.0 ...")
    send(300)
    con = db(path)
    first = check(con, thresholds)
    print("   check:", "no alerts" if not first else first)
    print("2. jev-latest now points at jev-1.13.0; 200 more tickets ...")
    state["model"] = "jev-1.13.0"
    send(200)
    print("   check:")
    notify(check(con, thresholds))
    up.shutdown(); px.shutdown()


# ---------------------------------------------------------------- self-check

def selftest():
    con = db(":memory:")
    rng = random.Random(1)
    req = {"questions": {"q": {"type": "noul", "instructions": "urgent?"}}}
    for _ in range(300):  # stable baseline and recent: no alerts
        record(con, "p", req, {"model": "m1", "answers": {"q": {"type": "noul", "noul": rng.betavariate(2, 8)}}}, 100)
    assert check(con, {"q": 0.7}) == [], check(con, {"q": 0.7})
    for _ in range(200):  # model switch that pushes answers onto the threshold
        record(con, "p", req, {"model": "m2", "answers": {"q": {"type": "noul", "noul": rng.uniform(0.66, 0.74)}}}, 100)
    kinds = {k for k, _, _ in check(con, {"q": 0.7})}
    assert kinds == {"version", "drift", "edge", "accuracy"}, kinds
    # an edited question is blamed on the edit, not the model
    con2 = db(":memory:")
    for i in range(400):
        spec = {"questions": {"q": {"type": "noul", "instructions": "urgent?" if i < 200 else "is it an emergency?"}}}
        record(con2, "p", spec, {"model": "m1", "answers": {"q": {"type": "noul", "noul": rng.betavariate(2, 8) if i < 200 else rng.betavariate(8, 2)}}}, 100)
    drift = [m for k, _, m in check(con2) if k == "drift"]
    assert drift and "question text changed" in drift[0], drift
    assert abs(psi(Counter(a=50, b=50), Counter(a=50, b=50))) < 1e-9
    # alerts fire once, stay quiet while they persist, resolve, and can fire again
    state, fired = db(":memory:"), [("drift", "p/q", "x"), ("edge", "p/q", "y")]
    assert len(track(state, fired)[0]) == 2
    assert track(state, fired) == ([], [])
    assert sorted(track(state, [])[1]) == [("drift", "p/q"), ("edge", "p/q")]
    assert len(track(state, fired)[0]) == 2
    try:
        import asyncio, httpx2
    except ImportError:
        httpx2 = None
    if httpx2:
        import tempfile
        wpath = os.path.join(tempfile.mkdtemp(prefix="tiltmeter-"), "w.db")
        canned = {"model": "jev-1.13.0", "answers": {"urgent": {"type": "noul", "noul": 0.9}}}
        mock = httpx2.MockTransport(lambda r: httpx2.Response(200, json=canned))

        async def roundtrip():
            async with instrument("w", wpath, transport=mock) as c:
                r = await c.post("https://api.typesafe.ai/v1/systemone",
                                 json={"questions": {"urgent": {"type": "noul", "instructions": "urgent?"}}})
                return r.json()
        assert asyncio.run(roundtrip()) == canned
        assert db(wpath).execute("select project, model, qid, value from answers").fetchall() == [("w", "jev-1.13.0", "urgent", 0.9)]
    print("ok  " + ("in-process wrapper records and passes answers through; " if httpx2 else "") + "stable traffic is quiet; a model switch fires version, drift, edge and accuracy; edits are blamed on the edit; alerts fire once until resolved")


def parse_thresholds(items):
    out = {}
    for s in items or []:
        k, v = s.split("=")
        out[k] = float(v)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["serve", "check", "demo", "selftest"])
    ap.add_argument("--port", type=int, default=4790)
    ap.add_argument("--upstream", default=UPSTREAM)
    ap.add_argument("--db", default=DB)
    ap.add_argument("--threshold", action="append", help="question_id=0.7, the value your code acts on")
    ap.add_argument("--webhook", help="Slack-compatible incoming webhook URL for alerts")
    ap.add_argument("--every", type=int, default=300, help="seconds between checks while serving")
    a = ap.parse_args()
    th = parse_thresholds(a.threshold)
    if a.cmd == "serve":
        serve(a.port, a.upstream, a.db, th, a.webhook, a.every)
    elif a.cmd == "check":
        con = db(a.db)
        new, resolved = track(con, check(con, th))
        notify(new, a.webhook, resolved)
        still = con.execute("select kind, name, msg from alerts where active=1").fetchall()
        print(f"{len(new)} new, {len(resolved)} resolved, {len(still)} active" + "".join(f"\n  active [{k}] {n}: {m}" for k, n, m in still))
    elif a.cmd == "demo":
        demo()
    else:
        selftest()
