"""H32 S3a step 7: word counts of the abstract and of the main text, written into VENUE_RULES.md.
Macros from generated/numbers.tex and generated/sentences.tex are expanded first (each number counts as one word),
then TeXcount (from the TinyTeX inside H32) counts words in text, headers and captions separately.
Main text = sections Introduction … Data and code availability (no abstract, acknowledgements, references, appendices).
    uv run python 07_manuscript/count_words.py
"""
import datetime, pathlib, re, subprocess

HERE = pathlib.Path(__file__).parent; ROOT = HERE.parent
TEXCOUNT = next((ROOT / ".tinytex/bin").glob("*/texcount"))
defs = {}
for f in ("generated/numbers.tex", "generated/sentences.tex"):
    for name, body in re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", (HERE / f).read_text(), re.M):
        defs[name] = body
src = (HERE / "main.tex").read_text()
for _ in range(3):
    src = re.sub(r"\\([A-Za-z]+)(?![A-Za-z])", lambda m: defs.get(m.group(1), m.group(0)) if m.group(1) in defs else m.group(0), src)
for name in ("T1", "T2", "T3", "T4", "T5"):                     # generated tables are counted as captions/tables, not text
    src = src.replace(f"\\input{{generated/{name}}}", (HERE / f"generated/{name}.tex").read_text())
abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", src, re.S).group(1)
body = re.search(r"(\\section\{Introduction\}.*?)\\section\*\{Acknowledgements\}", src, re.S).group(1)


def tc(text):
    p = HERE / "generated/_count.tex"; p.write_text("\\begin{document}\n" + text + "\n\\end{document}\n")
    out = subprocess.run([str(TEXCOUNT), "-1", "-sum=1,1,1", "-brief", "-q", str(p)], capture_output=True, text=True).stdout
    full = subprocess.run([str(TEXCOUNT), "-q", str(p)], capture_output=True, text=True).stdout
    p.unlink()
    g = lambda k: int(re.search(rf"{k}: (\d+)", full).group(1))
    return dict(text=g("Words in text"), headers=g("Words in headers"), captions=g("Words outside text \\(captions, etc.\\)"))


a_full = tc(abstract)
a = a_full                                               # T6 sentence written in (prompt S3d)
assert a["text"] <= 250, a
b = tc(body)
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
block = f"""<!-- WORDCOUNT:BEGIN -->
Counted {now} +07 by `07_manuscript/count_words.py` (TeXcount, macros expanded; numbers count as words).

| Part | Words | Journal limit | Source of the limit |
|---|---|---|---|
| Abstract, including the T6 sentence (headings included) | {a['text']} | "must be no longer than 250 words" | P2 |
| Main text, Introduction … Data and code availability: running text | {b['text']} | "not normally more than 8000 words" (Research paper) | P2 |
| Main text: section headings | {b['headers']} | — | — |
| Main text: captions and tables | {b['captions']} | — | — |

Appendices, references and acknowledgements are not counted.
<!-- WORDCOUNT:END -->"""
vr = (HERE / "VENUE_RULES.md").read_text()
vr = re.sub(r"<!-- WORDCOUNT:BEGIN -->.*?<!-- WORDCOUNT:END -->", lambda m: block, vr, flags=re.S)
(HERE / "VENUE_RULES.md").write_text(vr)
print("abstract", a, "body", b)
