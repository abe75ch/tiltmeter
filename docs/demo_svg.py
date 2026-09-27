"""Render `tiltmeter.py demo` output as an animated terminal SVG for the README.

  python3 docs/demo_svg.py   # writes docs/demo.svg
"""
import html, subprocess, sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = subprocess.run([sys.executable, str(root / "tiltmeter.py"), "demo"], capture_output=True, text=True, check=True).stdout

lines = [("cmd", "$ python3 tiltmeter.py demo")]
for raw in out.splitlines():
    if raw.startswith("ALERT"):
        head, _, msg = raw.partition(": ")
        lines += [("alert", head), ("msg", "    " + msg)]
    elif raw.strip():
        lines.append(("text", raw))

COLORS = {"cmd": "#7ee787", "text": "#c9d1d9", "alert": "#ff7b72", "msg": "#8b949e"}
STEP, HOLD, LH, PAD, W = 0.45, 6.0, 20, 18, 900
T = len(lines) * STEP + HOLD
H = PAD * 2 + 28 + LH * len(lines)
css, rows = [], []
for i, (kind, text) in enumerate(lines):
    at = i * STEP / T * 100
    css.append(f"@keyframes l{i}{{0%,{max(at - 0.01, 0):.2f}%{{opacity:0}}{at:.2f}%,97%{{opacity:1}}100%{{opacity:0}}}}"
               f".l{i}{{animation:l{i} {T:.1f}s infinite}}")
    y = PAD + 28 + LH * (i + 1) - 5
    rows.append(f'<text class="l{i}" x="{PAD}" y="{y}" fill="{COLORS[kind]}">{html.escape(text)}</text>')

svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Tiltmeter demo: a simulated Jev version switch fires alerts">
<style>text{{font:12.5px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;white-space:pre;opacity:0}}{''.join(css)}</style>
<rect width="{W}" height="{H}" rx="10" fill="#0d1117"/>
<circle cx="20" cy="18" r="6" fill="#ff5f56"/><circle cx="40" cy="18" r="6" fill="#ffbd2e"/><circle cx="60" cy="18" r="6" fill="#27c93f"/>
{chr(10).join(rows)}
</svg>
"""
(root / "docs" / "demo.svg").write_text(svg)
print(f"wrote docs/demo.svg: {len(lines)} lines, {T:.0f}s loop")
