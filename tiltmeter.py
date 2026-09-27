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

  tiltmeter serve [--port 4790] [--threshold is_urgent=0.7] [--webhook URL]
  tiltmeter check [--threshold support/is_urgent=0.7] [--webhook URL]
  tiltmeter demo          # simulated Jev, a model switch, alerts firing
  tiltmeter selftest

Point your client at http://127.0.0.1:4790/v1/systemone, or
http://127.0.0.1:4790/<project>/v1/systemone to keep apps apart. The TypeSafe SDK and
Pydantic AI both read TYPESAFE_BASE_URL, so setting it to http://127.0.0.1:4790 is enough.
Callers send their own API key; the proxy never adds one.

Or skip the proxy and record in-process with Pydantic AI (needs pydantic-ai-slim[typesafe]):

  from pydantic_ai.providers.typesafe import TypeSafeProvider
  import tiltmeter
  provider = TypeSafeProvider(http_client=tiltmeter.instrument(project="support"))
"""
import argparse, hashlib, json, math, os, random, sqlite3, sys, threading, time, urllib.error, urllib.parse, urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("TYPESAFE_ENDPOINT", "https://api.typesafe.ai")
DB = os.environ.get("TILTMETER_DB", "tiltmeter.db")
WINDOW = 200        # answers per window: recent vs the baseline before it
MIN_N = 50          # skip a question until each window has this many answers
PSI_ALERT = 0.2     # effect size: common rule of thumb, >0.1 moderate shift, >0.25 major
ALPHA = 0.001       # and the shift must be this unlikely under "nothing changed", so sample size and option count don't cause alarms
RARE = 10           # buckets with fewer answers than this across both windows are merged, keeping the chi-square approximation valid
Z_ALPHA = 3.09      # one-sided z for ALPHA, used by the accuracy-drop test
EDGE_BAND = 0.05    # "near the threshold" means within this distance
EDGE_ALERT = 0.2    # alert when this share of recent answers is near the threshold
ACC_DROP = 0.05     # alert when estimated accuracy falls by this much
HOP_BY_HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
              "proxy-authorization", "proxy-authenticate", "host", "content-length", "accept-encoding", "content-encoding"}


def db(path=None):
    con = sqlite3.connect(path or DB, timeout=10)
    con.execute("""create table if not exists answers(id integer primary key, ts real, project text, model text,
                   qid text, qhash text, qtype text, value real, choice text, conf real, latency_ms real)""")
    con.execute("create index if not exists answers_by_question on answers(project, qid, qtype, ts)")
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


def record_safely(db_path, project, req_bytes, resp_bytes, latency_ms, ts):
    """Recording must never break the caller: any failure is logged and dropped."""
    try:
        con = db(db_path)  # ponytail: one short-lived connection per call; pool it if write volume ever matters
        try:
            record(con, project, json.loads(req_bytes), json.loads(resp_bytes), latency_ms, ts=ts)
        finally:
            con.close()
    except Exception as e:  # noqa: BLE001 - deliberately broad, see docstring
        print(f"tiltmeter: record failed, answer not stored: {type(e).__name__}: {e}", file=sys.stderr)


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


def chi2_sf(x, df):
    """P(X >= x) for a chi-square variable: the regularized upper incomplete gamma Q(df/2, x/2)."""
    a, z = df / 2.0, x / 2.0
    if x <= 0:
        return 1.0
    if z < a + 1:  # series for the lower gamma, then take the complement
        term = total = 1.0 / a
        n = a
        while abs(term) > abs(total) * 1e-12:
            n += 1
            term *= z / n
            total += term
        return max(0.0, 1.0 - total * math.exp(-z + a * math.log(z) - math.lgamma(a)))
    b, c, d = z + 1 - a, 1e300, 1.0 / (z + 1 - a)  # Lentz continued fraction for the upper gamma
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        h *= d * c
        if abs(d * c - 1) < 1e-12:
            break
    return min(1.0, math.exp(-z + a * math.log(z) - math.lgamma(a)) * h)


def drift_test(base, recent):
    """(PSI, p-value). Under no change, PSI * n1*n2/(n1+n2) is approximately chi-square with k-1 df."""
    merged_b, merged_r = Counter(), Counter()
    for k in set(base) | set(recent):
        key = k if base.get(k, 0) + recent.get(k, 0) >= RARE else "_rare"
        merged_b[key] += base.get(k, 0)
        merged_r[key] += recent.get(k, 0)
    nb, nr, k = sum(merged_b.values()), sum(merged_r.values()), len(set(merged_b) | set(merged_r))
    if k < 2 or not nb or not nr:
        return 0.0, 1.0
    shift = psi(merged_b, merged_r)
    return shift, chi2_sf(shift * nb * nr / (nb + nr), k - 1)


def buckets(rows, qtype):
    if qtype == "noul":
        return Counter(min(int(v * 10), 9) for v, _, _ in rows)
    if qtype == "choice":
        return Counter(c for _, c, _ in rows)
    return Counter(round(v) for v, _, _ in rows)


def correct_probs(rows, qtype, t):
    """Per answer, the chance the decision is right if Jev is calibrated. A yes/no acted on at
    threshold t is right with p if p >= t, else 1 - p. A Choice is right with the chosen
    option's probability (not TypeSafe's 'confidence', which measures concentration).
    A Score is not a right-or-wrong decision, so it has none."""
    if qtype == "noul":
        return [v if v >= t else 1 - v for v, _, _ in rows]
    if qtype == "choice":
        return [v for v, _, _ in rows]
    return None


def mean_var(xs):
    m = sum(xs) / len(xs)
    return m, sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)


def majority(xs):
    return Counter(xs).most_common(1)[0][0]


def check(con, thresholds=None, window=WINDOW, min_n=MIN_N):
    """Thresholds are keyed by 'project/question' or just 'question' for every project."""
    thresholds, alerts = thresholds or {}, []
    groups = con.execute("select distinct project, qid, qtype from answers").fetchall()
    types_per_q = Counter((p, q) for p, q, _ in groups)
    for project, qid, qtype in groups:
        rows = con.execute("""select model, qhash, value, choice, conf from answers where project=? and qid=? and qtype=?
                              order by ts desc, id desc limit ?""", (project, qid, qtype, 2 * window)).fetchall()[::-1]
        recent, base = rows[-window:], rows[:-window]
        if len(base) < min_n or len(recent) < min_n:
            continue
        name = f"{project}/{qid}" + (f" ({qtype})" if types_per_q[(project, qid)] > 1 else "")
        bm, rm = majority(r[0] for r in base), majority(r[0] for r in recent)
        edited = majority(r[1] for r in base) != majority(r[1] for r in recent)
        cause = "question text changed" if edited else (f"model {bm} -> {rm}" if bm != rm else "inputs or model behaviour changed")
        if bm != rm:
            alerts.append(("version", name, f"model changed {bm} -> {rm}"))
        b, r = [x[2:] for x in base], [x[2:] for x in recent]
        shift, pval = drift_test(buckets(b, qtype), buckets(r, qtype))
        if shift > PSI_ALERT and pval < ALPHA:
            alerts.append(("drift", name, f"answer distribution shifted, PSI {shift:.2f} (p={pval:.1g}); likely cause: {cause}"))
        t = thresholds.get(f"{project}/{qid}", thresholds.get(qid))
        if qtype in ("noul", "choice") and t is not None:
            edge = sum(abs(v - t) < EDGE_BAND for v, _, _ in r) / len(r)  # yes-probability or chosen-option probability
            if edge > EDGE_ALERT:
                alerts.append(("edge", name, f"{edge:.0%} of recent answers within {EDGE_BAND} of threshold {t}: decisions are fragile"))
        t = 0.5 if t is None else t
        cb, cr = correct_probs(b, qtype, t), correct_probs(r, qtype, t)
        if cb is not None:
            (ab, vb), (ar, vr) = mean_var(cb), mean_var(cr)
            se = math.sqrt(vb / len(cb) + vr / len(cr))
            if ab - ar > ACC_DROP and (se == 0 or (ab - ar) / se > Z_ALPHA):
                alerts.append(("accuracy", name, f"estimated accuracy {ab:.0%} -> {ar:.0%} (no labels; assumes calibration); likely cause: {cause}"))
    return alerts


def track(con, alerts):
    """Work out which alerts are new and which resolved, without committing: the caller
    commits once the alerts are delivered, so a failed delivery is retried next check.
    An alert fires once and stays quiet while it persists; a version alert fires again
    if the model changes again."""
    con.execute("""create table if not exists alerts(kind text, name text, msg text, first_ts real, last_ts real,
                   active integer, primary key(kind, name))""")
    now, current = time.time(), {(k, n): m for k, n, m in alerts}
    active = {(k, n): m for k, n, m in con.execute("select kind, name, msg from alerts where active=1")}
    new = [(k, n, m) for (k, n), m in current.items() if (k, n) not in active or (k == "version" and active[(k, n)] != m)]
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
    return new, resolved


def notify(alerts, webhook=None, resolved=()):
    """Print alerts and post them to a Slack-compatible webhook. Returns False if delivery failed."""
    for kind, name, msg in alerts:
        print(f"ALERT [{kind}] {name}: {msg}", flush=True)
    for kind, name in resolved:
        print(f"RESOLVED [{kind}] {name}", flush=True)
    if not webhook or not (alerts or resolved):
        return True
    lines = [f"[{k}] {n}: {m}" for k, n, m in alerts] + [f"resolved [{k}] {n}" for k, n in resolved]
    body = json.dumps({"text": "\n".join(lines)}).encode()
    try:
        urllib.request.urlopen(urllib.request.Request(webhook, body, {"Content-Type": "application/json"}), timeout=10).read()
        return True
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"tiltmeter: webhook failed, alerts will be retried next check: {e}", file=sys.stderr)
        return False


def run_check(db_path, thresholds=None, webhook=None):
    """Check, deliver, then remember what was delivered. Returns (new, resolved, still_active)."""
    con = db(db_path)
    try:
        new, resolved = track(con, check(con, thresholds))
        if notify(new, webhook, resolved):
            con.commit()
        else:
            con.rollback()
        still = con.execute("select kind, name, msg from alerts where active=1").fetchall()
        return new, resolved, still
    finally:
        con.close()


# ---------------------------------------------------------------- in-process wrapper

def instrument(project="default", db_path=None, transport=None, **client_kwargs):
    """An httpx2.AsyncClient that records every Jev answer, for TypeSafeProvider(http_client=...)
    or AsyncTypeSafeClient. Your app gets the exact same responses, and a recording
    failure is logged, never raised."""
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
                    req_bytes = request.content
                except Exception:  # noqa: BLE001 - a streamed request body can't be re-read; skip recording
                    return response
                # ponytail: a blocking sqlite write of a few ms; move to a queue if it ever shows up in latency
                record_safely(path, project, req_bytes, body, (time.time() - start) * 1000, start)
            return response

        async def aclose(self):
            await self.inner.aclose()

    return httpx2.AsyncClient(transport=Recorder(), **client_kwargs)


# ---------------------------------------------------------------- proxy

def make_handler(upstream, db_path):
    class Proxy(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def reply(self, status, data, headers=()):
            self.send_response(status)
            for k, v in headers:
                if k.lower() not in HOP_BY_HOP:
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def fail(self, status, message):
            self.reply(status, json.dumps({"error": {"message": f"tiltmeter: {message}"}}).encode(),
                       [("Content-Type", "application/json")])

        def do_POST(self):
            url = urllib.parse.urlsplit(self.path)
            parts = [p for p in url.path.split("/") if p]
            if parts[-2:] != ["v1", "systemone"]:
                return self.fail(404, "use /v1/systemone or /<project>/v1/systemone")
            if not self.headers.get("Authorization"):
                return self.fail(401, "send your own TypeSafe API key in the Authorization header")
            if not (self.headers.get("Content-Type") or "").lower().startswith("application/json"):
                return self.fail(415, "Content-Type must be application/json")
            project = "/".join(parts[:-2]) or "default"
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            fwd = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
            target = f"{upstream}/v1/systemone" + (f"?{url.query}" if url.query else "")
            start = time.time()
            try:
                with urllib.request.urlopen(urllib.request.Request(target, body, fwd, method="POST"), timeout=60) as r:
                    status, data, headers = r.status, r.read(), r.headers.items()
            except urllib.error.HTTPError as e:
                status, data, headers = e.code, e.read(), e.headers.items()
            except TimeoutError:
                return self.fail(504, "TypeSafe did not answer within 60 s")
            except (urllib.error.URLError, OSError) as e:
                return self.fail(502, f"could not reach TypeSafe: {getattr(e, 'reason', e)}")
            latency = (time.time() - start) * 1000
            self.reply(status, data, headers)  # the caller gets Jev's status, headers and bytes
            if status == 200:
                record_safely(db_path, project, body, data, latency, start)  # request time, not write time
    return Proxy


def serve(port, upstream, db_path, thresholds, webhook, every=300):
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(upstream, db_path))
    print(f"tiltmeter proxy on http://127.0.0.1:{port}/v1/systemone -> {upstream}, checks every {every}s", flush=True)

    def loop():
        while True:
            time.sleep(every)
            try:
                run_check(db_path, thresholds, webhook)
            except Exception as e:  # noqa: BLE001 - keep monitoring alive; the next check retries
                print(f"tiltmeter: check failed, retrying in {every}s: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
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
            self.send_header("Content-Type", "application/json")
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
    headers = {"Content-Type": "application/json", "Authorization": "Bearer demo"}
    sent = [0]

    def send(n):
        for _ in range(n):
            urllib.request.urlopen(urllib.request.Request(url, body, headers)).read()
        sent[0] += n
        while db(path).execute("select count(*) from answers").fetchone()[0] < sent[0] * 2:
            time.sleep(0.05)  # the proxy saves answers after replying; wait for the last ones

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

def quiet():
    """Hide the alerts and warnings the self-check provokes on purpose."""
    import contextlib, io
    stack = contextlib.ExitStack()
    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
    stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
    return stack


def selftest():
    import tempfile
    tmp = tempfile.mkdtemp(prefix="tiltmeter-")
    con = db(":memory:")
    rng = random.Random(1)
    req = {"questions": {"q": {"type": "noul", "instructions": "urgent?"}}}
    for _ in range(300):  # stable baseline and recent: no alerts
        record(con, "p", req, {"model": "m1", "answers": {"q": {"type": "noul", "noul": rng.betavariate(2, 8)}}}, 100)
    assert check(con, {"q": 0.7}) == [], check(con, {"q": 0.7})
    for _ in range(200):  # model switch that pushes answers onto the threshold
        record(con, "p", req, {"model": "m2", "answers": {"q": {"type": "noul", "noul": rng.uniform(0.66, 0.74)}}}, 100)
    assert {k for k, _, _ in check(con, {"q": 0.7})} == {"version", "drift", "edge", "accuracy"}
    assert {k for k, _, _ in check(con, {"p/q": 0.7, "q": 0.1})} == {"version", "drift", "edge", "accuracy"}  # per-project wins
    # an edited question is blamed on the edit, not the model
    con2 = db(":memory:")
    for i in range(400):
        spec = {"questions": {"q": {"type": "noul", "instructions": "urgent?" if i < 200 else "is it an emergency?"}}}
        record(con2, "p", spec, {"model": "m1", "answers": {"q": {"type": "noul", "noul": rng.betavariate(2, 8) if i < 200 else rng.betavariate(8, 2)}}}, 100)
    drift = [m for k, _, m in check(con2) if k == "drift"]
    assert drift and "question text changed" in drift[0], drift
    # a field that changes type is tracked per type, never mixed into one window
    con3 = db(":memory:")
    for i in range(400):
        a = {"type": "noul", "noul": rng.betavariate(2, 8)} if i < 200 else {"type": "choice", "choice": "x", "probabilities": {"x": 0.9}, "confidence": 0.9}
        record(con3, "p", req, {"model": "m1", "answers": {"q": a}}, 1)
    assert check(con3) == [], check(con3)
    assert abs(psi(Counter(a=50, b=50), Counter(a=50, b=50))) < 1e-9
    for x, df in ((10.828, 1), (13.816, 2), (27.877, 9), (29.588, 10)):  # published 0.999 quantiles
        assert abs(chi2_sf(x, df) - 0.001) < 2e-5, (x, df, chi2_sf(x, df))
    assert abs(chi2_sf(2.0, 2) - math.exp(-1)) < 1e-9
    # steady traffic stays quiet whatever the sample size or the number of options
    def steady(kind, n_total, k=2, trials=60):
        fired = 0
        for seed in range(trials):
            g, c = random.Random(seed), db(":memory:")
            for _ in range(n_total):
                if kind == "noul":
                    a = {"type": "noul", "noul": g.betavariate(2, 5)}
                else:
                    o = f"o{g.randrange(k)}"
                    a = {"type": "choice", "choice": o, "probabilities": {o: g.uniform(0.5, 0.9)}, "confidence": 0.5}
                record(c, "p", req, {"model": "m", "answers": {"q": a}}, 1)
            fired += bool(check(c))
        return fired / trials
    assert steady("noul", 250) <= 0.05  # baseline of only 50 answers
    assert steady("choice", 400, k=30) <= 0.05 and steady("choice", 400, k=100) <= 0.05
    # while a moderate real shift is still caught
    c = db(":memory:")
    for i in range(400):
        record(c, "p", req, {"model": "m", "answers": {"q": {"type": "noul", "noul": rng.betavariate(2, 5) if i < 200 else rng.betavariate(4, 4)}}}, 1)
    assert "drift" in {k for k, _, _ in check(c)}
    # Choice accuracy uses the chosen option's probability, not confidence
    assert correct_probs([(0.7, "x", 0.55)], "choice", 0.5) == [0.7] and correct_probs([(1.2, None, 0.4)], "score", 0.5) is None
    # alerts fire once, stay quiet, resolve, fire again; a second model switch is new news
    state, fired = db(":memory:"), [("drift", "p/q", "x"), ("version", "p/q", "a -> b")]
    assert len(track(state, fired)[0]) == 2 and track(state, fired) == ([], [])
    assert track(state, [("drift", "p/q", "x"), ("version", "p/q", "b -> c")])[0] == [("version", "p/q", "b -> c")]
    assert sorted(track(state, [])[1]) == [("drift", "p/q"), ("version", "p/q")]
    assert len(track(state, fired)[0]) == 2
    # a failed webhook keeps alerts pending, so the next check delivers them
    wdb = os.path.join(tmp, "webhook.db")
    wcon = db(wdb)
    for i in range(400):
        record(wcon, "p", req, {"model": "m1" if i < 200 else "m2", "answers": {"q": {"type": "noul", "noul": 0.1 if i < 200 else 0.9}}}, 1)
    wcon.close()
    with quiet():
        assert len(run_check(wdb, webhook="http://127.0.0.1:9/down")[2]) == 0  # nothing remembered as delivered
        assert len(run_check(wdb)[0]) > 0  # so it is reported again

    # proxy: auth and content type required, headers and query pass through, dead upstream -> 502
    seen = {}

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            seen.update(auth=self.headers.get("Authorization"), extra=self.headers.get("X-Extra"), path=self.path)
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            data = json.dumps({"model": "m1", "answers": {"q": {"type": "noul", "noul": 0.4}}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("X-Typesafe-Request-Id", "req_123")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    up = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    pdb = os.path.join(tmp, "proxy.db")
    px = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(f"http://127.0.0.1:{up.server_port}", pdb))
    dead = ThreadingHTTPServer(("127.0.0.1", 0), make_handler("http://127.0.0.1:9", pdb))
    for s in (up, px, dead):
        threading.Thread(target=s.serve_forever, daemon=True).start()

    def post(server, headers, path="/app/v1/systemone"):
        r = urllib.request.Request(f"http://127.0.0.1:{server.server_port}{path}", json.dumps(req).encode(), headers)
        try:
            with urllib.request.urlopen(r, timeout=10) as resp:
                return resp.status, dict(resp.headers)
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers)

    good = {"Content-Type": "application/json", "Authorization": "Bearer k", "X-Extra": "1"}
    assert post(px, {"Content-Type": "application/json"})[0] == 401  # never lends the server's key
    assert post(px, {"Content-Type": "text/plain", "Authorization": "Bearer k"})[0] == 415  # blocks no-cors page requests
    status, headers = post(px, good, "/app/v1/systemone?trace=1")
    assert status == 200 and headers.get("X-Typesafe-Request-Id") == "req_123", headers
    assert seen == {"auth": "Bearer k", "extra": "1", "path": "/v1/systemone?trace=1"}, seen
    assert post(dead, good)[0] == 502
    for _ in range(50):
        if db(pdb).execute("select count(*) from answers").fetchone()[0]:
            break
        time.sleep(0.02)
    assert db(pdb).execute("select project, qid from answers").fetchall() == [("app", "q")]
    for s in (up, px, dead):
        s.shutdown()

    try:
        import asyncio, httpx2
    except ImportError:
        httpx2 = None
    if httpx2:
        wpath = os.path.join(tmp, "w.db")
        canned = {"model": "jev-1.13.0", "answers": {"urgent": {"type": "noul", "noul": 0.9}}}
        odd = {"model": "jev-1.13.0", "answers": {"urgent": None}}  # malformed: must not break the caller

        async def roundtrip(payload):
            mock = httpx2.MockTransport(lambda r: httpx2.Response(200, json=payload))
            async with instrument("w", wpath, transport=mock) as c:
                r = await c.post("https://api.typesafe.ai/v1/systemone",
                                 json={"questions": {"urgent": {"type": "noul", "instructions": "urgent?"}}})
                return r.json()
        assert asyncio.run(roundtrip(canned)) == canned
        with quiet():
            assert asyncio.run(roundtrip(odd)) == odd
        assert db(wpath).execute("select project, model, qid, value from answers").fetchall() == [("w", "jev-1.13.0", "urgent", 0.9)]
    print("ok  " + ("in-process wrapper passes answers through and never raises; " if httpx2 else "")
          + "proxy needs the caller's key, passes headers, returns 502 when TypeSafe is down; "
          + "alerts fire once, retry after a failed webhook, report a second model switch; drift needs significance, so false alarms stay low at any size; accuracy uses probabilities, not confidence")


def parse_thresholds(items):
    out = {}
    for s in items or []:
        k, v = s.rsplit("=", 1)
        out[k] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["serve", "check", "demo", "selftest"])
    ap.add_argument("--port", type=int, default=4790)
    ap.add_argument("--upstream", default=UPSTREAM)
    ap.add_argument("--db", default=DB)
    ap.add_argument("--threshold", action="append", help="question=0.7 or project/question=0.7, the value your code acts on")
    ap.add_argument("--webhook", help="Slack-compatible incoming webhook URL for alerts")
    ap.add_argument("--every", type=int, default=300, help="seconds between checks while serving")
    a = ap.parse_args()
    if a.webhook and urllib.parse.urlsplit(a.webhook).scheme not in ("http", "https"):
        ap.error("--webhook must be an http(s) URL, e.g. https://hooks.slack.com/services/...")
    th = parse_thresholds(a.threshold)
    if a.cmd == "serve":
        serve(a.port, a.upstream, a.db, th, a.webhook, a.every)
    elif a.cmd == "check":
        new, resolved, still = run_check(a.db, th, a.webhook)
        print(f"{len(new)} new, {len(resolved)} resolved, {len(still)} active" + "".join(f"\n  active [{k}] {n}: {m}" for k, n, m in still))
    elif a.cmd == "demo":
        demo()
    else:
        selftest()


if __name__ == "__main__":
    main()
