"""Embed the year-18 inspection (seed_observations.json) into the single-file app."""
import json, pathlib
here = pathlib.Path(__file__).parent
seed = json.load(open(here / "seed_observations.json"))
for o in seed:
    o.setdefault("extent", "local"); o.setdefault("inspector", "S. (year-18 detailed inspection)"); o.setdefault("date", "")
html = (here / "template.html").read_text(encoding="utf-8")
html = html.replace("/*SEED*/[]", json.dumps(seed, separators=(",", ":")))
(here / "index.html").write_text(html, encoding="utf-8")
print(f"index.html written with {len(seed)} seed observations")
