"""H32 S3e step 5: check the references of refs.bib against doi.org and Crossref (tool output only; refs.bib is not
changed). For each entry with a DOI: HTTP status and Location of `curl -sI https://doi.org/<DOI>`; Crossref
https://api.crossref.org/works/<DOI> fields title, container-title, issued, volume, page / article-number; compared
with refs.bib field by field (case, braces, LaTeX accents, dashes and HTML tags ignored). Entries without a DOI: URL and
HTTP status. A generic User-Agent is sent (no personal data).
    uv run python 07_manuscript/ref_check.py      -> 07_manuscript/REF_CHECK.md
"""
import datetime, json, pathlib, re, subprocess, time, unicodedata

HERE = pathlib.Path(__file__).parent
UA = "H32-refcheck/1.0"
REQUIRED = [("Sim2026", "Sim & Kim 2026"), ("ElAllam2026Meas", "El Allam & Hamlich 2026 (Measurement)"),
            ("ElAllam2026BSPC", "El Allam & Hamlich 2026 (BSPC)"), ("ElAllam2026PM", "El Allam & Hamlich 2026 (Physiol. Meas.)"),
            ("Kinalioglu2026", "Kinalioglu 2026"), ("DiazPulido2026", "Díaz-Pulido et al. 2026"),
            ("YangLi2025", "Yang & Li 2025"), ("Jui2026", "Jui et al. 2026")]
NODOI = ["Oliveira2024", "GibbsCandes2021"]

INTRO = """## 5. Intro sentence (prompt S3f, A1; full texts re-read 2026-11-24)

Sentence in main.tex (Introduction, paragraph 1), citing only entries already in refs.bib:

> Wrist photoplethysmography (PPG) moves heart-rate monitoring out of the clinic, where motion and environment change
> over the day and degrade heart-rate estimation (Reiss et al., 2019; Meier et al., 2024). We are not aware of a
> published estimate of the false-alarm rate of wrist-PPG heart-rate alarms under free-living wear.

| Entry | Full text read | Verbatim support | Location | Supports |
|---|---|---|---|---|
| `Reiss2019` | PMC full-text XML PMC6679242 (`01_lit/pdf/Reiss2019_PMC6679242.xml`, local, not committed). The MDPI PDF returned HTTP 403 and the PMC PDF link returned an HTML page (2026-11-24), so **no page numbers are available**; section numbers are given | "However, compared to the traditional ECG data, heart rate estimation is more challenging based on the PPG-signal. Especially motion artefacts, caused by the user’s physical activity, can significantly reduce the PPG-signal quality. Therefore, increasing the accuracy and robustness of PPG-based heart rate estimation remains at the forefront of research in this area." Also: "This observation was expected, since an elevated heart rate usually corresponds to increased physical activity, leading to larger motion artefacts (for example, the deep learning approach has an MAE of 16.98 bpm during the activity stairs)." | section 1 (Introduction); section 4.3 | motion degrades HR estimation: **yes**. Environment: not stated |
| `Meier2024` | arXiv:2412.17540 PDF, downloaded 2026-11-24 from https://arxiv.org/pdf/2412.17540 (21 pages; scratch copy, not committed); text extracted with macOS PDFKit | "However, obtaining reliable HR estimations from PPG signals in real-world conditions and during real-world activities is challenging. Several external factors negatively impact accurate estimation, such as motion artifacts [5] that arise from a person’s movements and performed activities [6, 7], sensor misplacement [8] or slippage during wear, and environmental conditions that change over time such as low temperatures [9] or high levels of ambient light [10]." | p. 1 (PDF page = printed page of this version; the bibliography cites the NeurIPS proceedings version) | motion: **yes**; environment changing over time: **yes** |

Both entries support the sentence (Reiss2019 for motion, Meier2024 for motion and environment); neither was removed.
The second sentence ("We are not aware of …") is a statement about our search, not a citation."""

RECHECK = """## 6. Re-checks of single fields (prompt S3f, A2–A3, 2026-11-24)

| Entry | Check | Result | Action |
|---|---|---|---|
| `Jui2026` | Crossref `volume`, `issue`, `page` | volume none, issue none, page "1-1" (as returned) | refs.bib unchanged: note "Early access" kept |
| `Schmid1995` | Crossref `page` / `article-number`; publisher page behind doi.org; publisher PDF | Crossref: page none, article-number "111"; `https://doi.org/10.1007/bf02926025` → 302 → link.springer.com, which returned a bot challenge page ("Client Challenge"), no page range; publisher PDF `01_lit/pdf/schmid1995.pdf` page 1 header: "Statistical Papers 36, 111 - 130 (1995)", 20 pages numbered 111–130 | pages = 111--130 added via `PAGE_FIX` in `01_lit/make_bib.py` (source stated there). Table 3 above therefore shows the page field as different from Crossref's "111" |"""

bib = (HERE / "refs.bib").read_text()
E = {}
for m in re.finditer(r"@(\w+)\{([^,]+),\n(.*?)\n\}", bib, re.S):
    E[m.group(2)] = dict(re.findall(r"^\s*(\w+) = \{(.*)\},?$", m.group(3), re.M)); E[m.group(2)]["_type"] = m.group(1)

ACC = {r"{\`e}": "è", r"{\'e}": "é", r"{\~a}": "ã", r"{\"o}": "ö", r"{\"u}": "ü", r"{\'a}": "á", r"{\'i}": "í", r"{\'o}": "ó"}


def norm(s):
    s = str(s or "")
    for a, b in ACC.items():
        s = s.replace(a, b)
    s = re.sub(r"<[^>]+>", "", s).replace("{", "").replace("}", "").replace("\\&", "&").replace("&amp;", "&")
    s = unicodedata.normalize("NFC", s).replace("–", "-").replace("—", "-").replace("--", "-").replace("‐", "-")
    return re.sub(r"\s+", " ", s).strip().lower()


def curl_head(url):
    out = subprocess.run(["curl", "-sI", "-A", UA, "--max-time", "30", url], capture_output=True, text=True).stdout
    code = re.search(r"^HTTP/\S+ (\d+)", out, re.M); loc = re.search(r"^location: (\S+)", out, re.M | re.I)
    return (code.group(1) if code else "none"), (loc.group(1) if loc else "")


def crossref(doi):
    out = subprocess.run(["curl", "-s", "-A", UA, "--max-time", "30", f"https://api.crossref.org/works/{doi}"], capture_output=True, text=True).stdout
    try:
        return json.loads(out)["message"]
    except Exception:
        return None


def cmp(field, b, c):                            # -> (text, ok); missing on either side is ok but listed
    if b is None or b == "":
        return (f"{field} not in refs.bib (Crossref: {c})" if c else ""), True
    if c is None or c == "":
        return f"{field} not in Crossref (refs.bib: {b})", True
    ok = norm(b) == norm(c)
    return (f"{field}: {'KHỚP' if ok else 'LỆCH'}" + ("" if ok else f" (refs.bib: {b} | Crossref: {c})")), ok


rows = []; detail = []
keys = [k for k, _ in REQUIRED] + sorted(k for k in E if "doi" in E[k] and k not in dict(REQUIRED))
for key in keys:
    e = E[key]; doi = e["doi"]; code, loc = curl_head(f"https://doi.org/{doi}"); time.sleep(0.3)
    m = crossref(doi); time.sleep(0.3)
    if m is None:
        rows.append((key, doi, code, loc, "no Crossref record", "—", "—")); continue
    issued = "-".join(str(x) for x in (m.get("issued", {}).get("date-parts") or [[None]])[0] if x is not None)
    ct = (m.get("container-title") or [""])[0]
    page = m.get("article-number") or m.get("page") or ""
    checks = [cmp("title", e.get("title"), (m.get("title") or [""])[0]),
              cmp("container", e.get("journal") or e.get("booktitle") or e.get("howpublished"), ct),
              cmp("year", e.get("year"), issued[:4]),
              cmp("volume", e.get("volume"), m.get("volume")),
              cmp("pages", e.get("pages"), page)]
    core_ok = all(ok for (_, ok), f in zip(checks, ("title", "container", "year", "volume", "pages")) if f in ("title", "year"))
    all_ok = all(ok for _, ok in checks)
    verdict = "KHỚP" if all_ok else ("KHỚP (title, year); khác ở trường khác" if core_ok else "LỆCH")
    rows.append((key, doi, code, loc, verdict, "; ".join(t for t, ok in checks if not ok) or "—",
                 "; ".join(t for t, ok in checks if ok and " not in " in t) or "—"))
    detail.append((key, doi, (m.get("title") or [""])[0], ct, issued, m.get("volume"), page))
nd = []
for key in NODOI:
    e = E[key]; code, loc = curl_head(e["url"]); nd.append((key, e["url"], code, loc))

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
L = ["# H32 · Reference check (prompt S3e, step 5)", "",
     f"Generated {now} +07 by `07_manuscript/ref_check.py` from tool output (`curl -sI https://doi.org/<DOI>`; Crossref REST API "
     "`https://api.crossref.org/works/<DOI>`). refs.bib was **not** changed. Comparison ignores case, braces, LaTeX accents, "
     "dashes and HTML tags. Verdict KHỚP = title, container, year, volume and pages equal wherever both sides have the field (a field present on one side only is listed, not counted as a difference); 'KHỚP (title, year)' = the "
     "work is the same but another field differs or is missing (listed); LỆCH = title or year differs.", "",
     "## 1. The 8 required entries", "", "| # | Entry | Key | DOI | doi.org HTTP | Location | Verdict | Fields not equal | Field missing on one side |", "|---|---|---|---|---|---|---|---|---|"]
req = dict(REQUIRED)
for i, r in enumerate([r for r in rows if r[0] in req], 1):
    L.append(f"| {i} | {req[r[0]]} | `{r[0]}` | `{r[1]}` | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |")
L += ["", "## 2. Entries without a DOI", "", "| Key | URL | HTTP | Location |", "|---|---|---|---|"]
L += [f"| `{k}` | {u} | {c} | {l or '—'} |" for k, u, c, l in nd]
L += ["", "## 3. All other entries with a DOI", "", "| Key | DOI | doi.org HTTP | Verdict | Fields not equal | Field missing on one side |", "|---|---|---|---|---|---|"]
L += [f"| `{r[0]}` | `{r[1]}` | {r[2]} | {r[4]} | {r[5]} | {r[6]} |" for r in rows if r[0] not in req]
L += ["", "## 4. Crossref values as returned (title, container-title, issued, volume, page/article-number)", "",
      "| Key | Title | Container | Issued | Volume | Page |", "|---|---|---|---|---|---|"]
L += [f"| `{k}` | {re.sub(r'<[^>]+>', '', t).replace('|', '/')} | {c} | {i} | {v or '—'} | {p or '—'} |" for k, d, t, c, i, v, p in detail]
n_doi = len(rows); n_ok = sum(r[4] == "KHỚP" for r in rows); n_core = sum(r[4].startswith("KHỚP (") for r in rows); n_bad = sum(r[4] in ("LỆCH", "no Crossref record") for r in rows)
L += ["", f"Summary: {n_doi} entries with a DOI; KHỚP {n_ok}; KHỚP (title, year) with another field different {n_core}; "
      f"LỆCH or no Crossref record {n_bad}. doi.org codes: " + ", ".join(f"{c} × {sum(r[2] == c for r in rows)}" for c in sorted({r[2] for r in rows})) + "."]
L += ["", INTRO, "", RECHECK]
(HERE / "REF_CHECK.md").write_text("\n".join(L) + "\n")
print(L[-1])
