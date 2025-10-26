# # paper_patent_fit_module.py
# from __future__ import annotations
# import os, re, json, html, hashlib, asyncio, pathlib
# from pathlib import Path
# from typing import Dict, Any, List, Tuple, Optional
# import pandas as pd

# # =========================
# # Config / Constants
# # =========================
# FAST_MODEL = "gpt-4o-mini"  # fast default
# STRONG_MODEL = "gpt-4o"     # used in optional two-stage rescoring
# CACHE_DIR = Path(".gpt_cache")
# CACHE_DIR.mkdir(exist_ok=True)

# # Truncation (huge latency/tokens win)
# MAX_TITLE_CH = 220
# MAX_ABS_CH   = 1200

# # =========================
# # API key / client handling
# # =========================
# def load_api_key(path: str | Path = "secrets/openai_key.txt") -> None:
#     """
#     Ensure OPENAI_API_KEY is set. If missing, read a one-line key from `path`.
#     No-op if already set.
#     """
#     if os.getenv("OPENAI_API_KEY"):
#         return
#     p = Path(path)
#     if p.exists():
#         key = p.read_text(encoding="utf-8").strip()
#         if key:
#             os.environ["OPENAI_API_KEY"] = key

# def _get_client(model: str = FAST_MODEL):
#     """
#     Return (client, model). If import/env fails, returns (None, None) for mock mode.
#     """
#     try:
#         from openai import OpenAI
#         if not os.getenv("OPENAI_API_KEY"):
#             load_api_key()
#         client = OpenAI()
#         return client, model
#     except Exception:
#         return None, None

# def _get_async_client(model: str = FAST_MODEL):
#     try:
#         from openai import AsyncOpenAI
#         if not os.getenv("OPENAI_API_KEY"):
#             load_api_key()
#         client = AsyncOpenAI()
#         return client, model
#     except Exception:
#         return None, None

# # =========================
# # Small helpers
# # =========================
# def _norm(s: str) -> str:
#     return (s or "").strip()

# def _split_names(s: str) -> List[str]:
#     if not isinstance(s, str): return []
#     parts = re.split(r"[;,]", s)
#     return [p.strip() for p in parts if p.strip()]

# def _overlap_names(paper_first: str, paper_last: str, inventors: List[str]) -> Dict[str, Any]:
#     authors = [a for a in [_norm(paper_first), _norm(paper_last)] if a]
#     al, il = {a.lower() for a in authors}, {i.lower() for i in inventors}
#     overlap = sorted(al.intersection(il))
#     return {"overlap": bool(overlap), "overlap_names": overlap, "authors": authors, "inventors": inventors}

# def _safe_int(x, lo=1, hi=10) -> int:
#     try:
#         v = int(x); return max(lo, min(hi, v))
#     except Exception:
#         return 0

# def _clip(s: str, n: int) -> str:
#     s = (s or "").strip()
#     return (s[:n] + "…") if len(s) > n else s

# # =========================
# # Caching (prompt → JSON)
# # =========================
# def _prompt_key(prompt: str, model: str) -> Path:
#     h = hashlib.sha256((model + "\n" + prompt).encode("utf-8")).hexdigest()
#     return CACHE_DIR / f"{h}.json"

# def _read_cache(prompt: str, model: str) -> Optional[dict]:
#     p = _prompt_key(prompt, model)
#     if p.exists():
#         try:
#             return json.loads(p.read_text("utf-8"))
#         except Exception:
#             return None
#     return None

# def _write_cache(prompt: str, model: str, obj: dict) -> None:
#     _prompt_key(prompt, model).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")

# # =========================
# # Prompting & calls
# # =========================
# def _make_pair_prompt(paper: pd.Series, patent: pd.Series, intent: str) -> Tuple[str, Dict[str, Any]]:
#     first_author = _norm(paper.get("first_author", ""))
#     last_author  = _norm(paper.get("last_author",  ""))
#     inventors    = _split_names(_norm(patent.get("lead_inventor", "")))

#     ov   = _overlap_names(first_author, last_author, inventors)
#     doi  = _norm(paper.get("doi", ""))
#     pdf  = _norm(paper.get("pdf_url", ""))
#     pubn = _norm(patent.get("publication_number", ""))

#     paper_title   = _clip(paper.get('title',''), MAX_TITLE_CH)
#     paper_abs     = _clip(paper.get('abstract',''), MAX_ABS_CH)
#     patent_title  = _clip(patent.get('title',''), MAX_TITLE_CH)
#     patent_abs    = _clip(patent.get('abstract',''), MAX_ABS_CH)

#     prompt = f"""
# You are a senior venture & IP analyst. Goal: **{intent}**.

# Compare the RESEARCH PAPER and the PATENT and output strict JSON with keys:
# - score (int 1..10)
# - summary (string, 1–3 sentences)
# - ranked_authors: array of {{name, score_1to10, reason}} sorted best-first
# - ranked_inventors: array of {{name, score_1to10, reason}} sorted best-first
# - overlap_bonus_applied (boolean)

# If any paper author also appears among inventors, apply a positive but defensible bias.

# RESEARCH PAPER
# - DOI: {doi}
# - Title: {paper_title}
# - Authors: {first_author} ; {last_author}
# - Citations: {paper.get('citations_total','')}
# - Abstract: {paper_abs}
# - PDF: {pdf}

# PATENT
# - Publication #: {pubn}
# - Title: {patent_title}
# - Lead Inventor(s): {', '.join(inventors) if inventors else '(none)'}
# - Abstract: {patent_abs}

# Name-overlap hint (precomputed):
# - overlap: {ov['overlap']}
# - overlap_names: {ov['overlap_names']}
# """
#     return prompt.strip(), ov

# def _call_gpt_json(prompt: str, model: str = FAST_MODEL) -> Dict[str, Any]:
#     """
#     Structured JSON via Chat Completions JSON mode (sync).
#     Uses disk cache; falls back to mock if client/model unavailable.
#     """
#     cached = _read_cache(prompt, model)
#     if cached:
#         return cached

#     client, m = _get_client(model)
#     if client is None or m is None:
#         return {"score": 5, "summary": "(mock) average fit.",
#                 "ranked_authors": [], "ranked_inventors": [], "overlap_bonus_applied": False}

#     system = ("You are precise. Respond with strict JSON only containing: "
#               "score, summary, ranked_authors, ranked_inventors, overlap_bonus_applied.")
#     resp = client.chat.completions.create(
#         model=m,
#         response_format={"type": "json_object"},
#         messages=[{"role": "system", "content": system},
#                   {"role": "user", "content": prompt}],
#         temperature=0,
#         max_tokens=400,
#     )
#     raw = resp.choices[0].message.content.strip()
#     try:
#         obj = json.loads(raw)
#     except Exception:
#         mobj = re.search(r"\{.*\}\s*$", raw, flags=re.DOTALL)
#         obj = json.loads(mobj.group(0)) if mobj else {
#             "score": 0, "summary": raw, "ranked_authors": [], "ranked_inventors": [], "overlap_bonus_applied": False
#         }
#     _write_cache(prompt, model, obj)
#     return obj

# # Async variant (for speed via parallelism)
# async def _acall_gpt_json(prompt: str, model: str = FAST_MODEL) -> Dict[str, Any]:
#     cached = _read_cache(prompt, model)
#     if cached:
#         return cached

#     aclient, m = _get_async_client(model)
#     if aclient is None or m is None:
#         return {"score": 5, "summary": "(mock) average fit.",
#                 "ranked_authors": [], "ranked_inventors": [], "overlap_bonus_applied": False}

#     system = ("You are precise. Respond with strict JSON only containing: "
#               "score, summary, ranked_authors, ranked_inventors, overlap_bonus_applied.")
#     resp = await aclient.chat.completions.create(
#         model=m,
#         response_format={"type": "json_object"},
#         messages=[{"role":"system","content":system},
#                   {"role":"user","content":prompt}],
#         temperature=0,
#         max_tokens=400,
#     )
#     raw = resp.choices[0].message.content.strip()
#     try:
#         obj = json.loads(raw)
#     except Exception:
#         mobj = re.search(r"\{.*\}\s*$", raw, flags=re.DOTALL)
#         obj = json.loads(mobj.group(0)) if mobj else {
#             "score": 0, "summary": raw, "ranked_authors": [], "ranked_inventors": [], "overlap_bonus_applied": False
#         }
#     _write_cache(prompt, model, obj)
#     return obj

# # =========================
# # Prefilter (cheap pruning)
# # =========================
# def prefilter_pairs(papers_df: pd.DataFrame, patents_df: pd.DataFrame,
#                     top_papers: int = 15, top_patents: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
#     """Reduce cartesian product size quickly (adjust thresholds to taste)."""
#     if "citations_total" in papers_df.columns:
#         papers_df = papers_df.sort_values("citations_total", ascending=False).head(top_papers)
#     else:
#         papers_df = papers_df.head(top_papers)
#     patents_df = patents_df.head(top_patents)
#     return papers_df, patents_df

# # =========================
# # Scoring (async + sync wrapper)
# # =========================
# async def score_pairs_async(
#     papers_df: pd.DataFrame,
#     patents_df: pd.DataFrame,
#     intent: str = "Find market fit and propose a founding team",
#     model: str = FAST_MODEL,
#     concurrency: int = 6,
# ) -> pd.DataFrame:
#     sem = asyncio.Semaphore(concurrency)
#     tasks: List[asyncio.Task] = []

#     async def worker(p: pd.Series, t: pd.Series):
#         async with sem:
#             prompt, ov = _make_pair_prompt(p, t, intent)
#             obj = await _acall_gpt_json(prompt, model)
#             ra = obj.get("ranked_authors", []) or []
#             ri = obj.get("ranked_inventors", []) or []
#             return {
#                 "paper_doi": p.get("doi",""),
#                 "paper_title": p.get("title",""),
#                 "paper_first_author": p.get("first_author",""),
#                 "paper_last_author": p.get("last_author",""),
#                 "paper_citations": p.get("citations_total",""),
#                 "patent_id": t.get("publication_number",""),
#                 "patent_title": t.get("title",""),
#                 "patent_inventor": t.get("lead_inventor",""),
#                 "fit_score": _safe_int(obj.get("score")),
#                 "summary": obj.get("summary",""),
#                 "top_author": (ra[0]["name"] if ra else ""),
#                 "top_author_fit": (ra[0].get("score_1to10") if ra else ""),
#                 "top_inventor": (ri[0]["name"] if ri else ""),
#                 "top_inventor_fit": (ri[0].get("score_1to10") if ri else ""),
#                 "overlap_bonus_applied": bool(obj.get("overlap_bonus_applied", False) or ov["overlap"]),
#                 "ranked_authors_json": json.dumps(ra, ensure_ascii=False),
#                 "ranked_inventors_json": json.dumps(ri, ensure_ascii=False),
#             }

#     for _, p in papers_df.iterrows():
#         for _, t in patents_df.iterrows():
#             tasks.append(asyncio.create_task(worker(p, t)))

#     rows = await asyncio.gather(*tasks)
#     df = pd.DataFrame(rows)
#     df["fit_score"] = pd.to_numeric(df["fit_score"], errors="coerce")
#     return df.sort_values("fit_score", ascending=False, na_position="last")

# def score_pairs(
#     papers_df: pd.DataFrame,
#     patents_df: pd.DataFrame,
#     intent: str = "Find market fit and propose a founding team",
#     model: str = FAST_MODEL,
#     concurrency: int = 6,
# ) -> pd.DataFrame:
#     """Sync facade; runs the async scorer with concurrency."""
#     try:
#         return asyncio.run(score_pairs_async(papers_df, patents_df, intent, model, concurrency))
#     except RuntimeError:
#         # Already in an event loop (e.g., Jupyter/ASGI). Use loop directly.
#         loop = asyncio.get_event_loop()
#         return loop.run_until_complete(score_pairs_async(papers_df, patents_df, intent, model, concurrency))

# # =========================
# # Executive summary
# # =========================
# def summarize_top_pairs(
#     top_df: pd.DataFrame,
#     intent: str = "Find market fit and propose a founding team",
#     model: str = FAST_MODEL,
# ) -> Dict[str, Any]:
#     client, m = _get_client(model)
#     if client is None or m is None:
#         recs = []
#         for i, r in enumerate(top_df.itertuples(index=False), 1):
#             recs.append({"rank": i, "paper_title": getattr(r,"paper_title",""),
#                          "patent_title": getattr(r,"patent_title",""),
#                          "reason": f"Fit {getattr(r,'fit_score','')}; overlap={getattr(r,'overlap_bonus_applied',False)}."})
#         return {"summary": f"Top {len(recs)} pairs by fit for: {intent}.", "recommendations": recs}

#     items = []
#     for i, r in enumerate(top_df.itertuples(index=False), 1):
#         items.append({
#             "rank": i,
#             "paper_title": getattr(r,"paper_title",""),
#             "paper_doi": getattr(r,"paper_doi",""),
#             "patent_title": getattr(r,"patent_title",""),
#             "patent_id": getattr(r,"patent_id",""),
#             "fit_score": int(getattr(r,"fit_score",0) or 0),
#             "overlap_bonus": bool(getattr(r,"overlap_bonus_applied",False)),
#             "top_author": getattr(r,"top_author",""),
#             "top_inventor": getattr(r,"top_inventor",""),
#         })

#     system = ("You are a concise venture/IP analyst. Write strict JSON with keys: "
#               "'summary' (3–5 sentences) and 'recommendations' "
#               "(array of {rank,paper_title,patent_title,reason}, 1–2 lines each).")
#     user = {"intent_task": intent, "top_pairs": items}

#     resp = client.chat.completions.create(
#         model=m,
#         response_format={"type":"json_object"},
#         messages=[{"role":"system","content":system},
#                   {"role":"user","content":json.dumps(user, ensure_ascii=False)}],
#         temperature=0,
#         max_completion_tokens=400
#     )
#     raw = resp.choices[0].message.content.strip()
#     try:
#         return json.loads(raw)
#     except Exception:
#         return {"summary": f"Top {len(items)} by fit.",
#                 "recommendations": [
#                     {"rank": it["rank"], "paper_title": it["paper_title"], "patent_title": it["patent_title"],
#                      "reason": f"Fit {it['fit_score']}; overlap={it['overlap_bonus']}."} for it in items]}

# # =========================
# # HTML rendering
# # =========================
# def _linkify(url: str, text: str) -> str:
#     t = html.escape(text or "link")
#     u = (url or "").strip()
#     return f'<a href="{html.escape(u, quote=True)}" target="_blank" rel="noopener noreferrer">{t}</a>' if u else t

# def _details(text: str) -> str:
#     if not isinstance(text, str) or not text.strip(): return ""
#     short = html.escape(text[:180]) + ("..." if len(text) > 180 else "")
#     full  = html.escape(text)
#     return f'<details><summary>{short}</summary><div style="white-space:pre-wrap;margin-top:.5rem;">{full}</div></details>'

# def _pre(json_text: str) -> str:
#     if not json_text: return ""
#     return f'<details><summary>show</summary><pre style="white-space:pre-wrap;margin:.5rem 0;">{html.escape(json_text)}</pre></details>'

# def render_matches_html(
#     df: pd.DataFrame,
#     exec_summary: Optional[Dict[str, Any]] = None,
#     table_id: str = "matches_table"
# ) -> str:
#     dfx = df.copy()
#     dfx["fit_score"] = pd.to_numeric(dfx["fit_score"], errors="coerce")
#     anchors = [f"row-{i}" for i in range(1, len(dfx)+1)]

#     table = pd.DataFrame({
#         "": [f'<a id="{aid}"></a>' for aid in anchors],
#         "#": range(1, len(dfx)+1),
#         "Paper": [_linkify(dfx.get("paper_doi","").iloc[i], dfx.get("paper_title","").iloc[i]) for i in range(len(dfx))],
#         "First Author": dfx.get("paper_first_author",""),
#         "Last Author": dfx.get("paper_last_author",""),
#         "Citations": dfx.get("paper_citations",""),
#         "Patent ID": dfx.get("patent_id",""),
#         "Patent Title": dfx.get("patent_title",""),
#         "Lead Inventor": dfx.get("patent_inventor",""),
#         "Fit Score": dfx["fit_score"].map(lambda x: f"{x:.0f}" if pd.notna(x) else ""),
#         "Summary": dfx.get("summary","").map(_details),
#         "Top Author (fit)": dfx.apply(lambda r: f"{r.get('top_author','')} ({r.get('top_author_fit','')})", axis=1),
#         "Top Inventor (fit)": dfx.apply(lambda r: f"{r.get('top_inventor','')} ({r.get('top_inventor_fit','')})", axis=1),
#         "Overlap Bonus?": dfx.get("overlap_bonus_applied", False).map(lambda x: "Yes" if x else "No"),
#         "Ranked Authors": dfx.get("ranked_authors_json","").map(_pre),
#         "Ranked Inventors": dfx.get("ranked_inventors_json","").map(_pre),
#     })
#     table_html = table.to_html(index=False, escape=False, border=0, table_id=table_id, classes="display nowrap compact")

#     summary_html = ""
#     if exec_summary:
#         items = ""
#         for rec in exec_summary.get("recommendations", []):
#             rid = int(rec.get("rank", 0))
#             items += (
#                 f'<li><a href="#row-{rid}"><strong>#{rid}</strong> '
#                 f'{html.escape(rec.get("paper_title",""))} ↔ {html.escape(rec.get("patent_title",""))}</a>'
#                 f'<br><span style="color:#555;">{html.escape(rec.get("reason",""))}</span></li>'
#             )
#         summary_html = f"""
#         <div class="card" style="margin-bottom:1rem;">
#           <div class="card-body">
#             <h3 class="card-title">Executive summary</h3>
#             <p>{html.escape(exec_summary.get("summary",""))}</p>
#             <h5>Top recommendations</h5>
#             <ol>{items}</ol>
#           </div>
#         </div>
#         """

#     cols_json = json.dumps(list(table.columns))
#     return f"""<!DOCTYPE html>
# <html>
# <head>
#   <meta charset="utf-8"/>
#   <title>Paper–Patent Matches (Top-K)</title>
#   <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"/>
#   <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css"/>
#   <style>
#     body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:1.5rem;background:#fafafa}}
#     table.dataTable thead th{{white-space:nowrap}} details>summary{{cursor:pointer;color:#0d6efd}}
#     details[open]>summary{{color:#0a58ca}} .card{{box-shadow:0 2px 10px rgba(0,0,0,.05)}}
#   </style>
# </head>
# <body>
#   {summary_html}
#   <h2>Top Matches (Paper ↔ Patent) with Author/Inventor Ranking</h2>
#   {table_html}
#   <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
#   <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
#   <script>
#     $(function(){{
#       const names={cols_json}; const scoreIdx=names.indexOf("Fit Score");
#       $('#{table_id}').DataTable({{paging:true,searching:true,order:scoreIdx>=0?[[scoreIdx,'desc']]:[[0,'asc']],scrollX:true,responsive:true,pageLength:10,lengthMenu:[10,25,50,100]}});
#     }});
#   </script>
# </body>
# </html>"""

# # =========================
# # Two-stage rescoring (optional)
# # =========================
# def score_pairs_two_stage(
#     papers_df: pd.DataFrame,
#     patents_df: pd.DataFrame,
#     intent: str,
#     fast_model: str = FAST_MODEL,
#     strong_model: str = STRONG_MODEL,
#     top_m: int = 15,
#     concurrency: int = 6,
# ) -> pd.DataFrame:
#     """Fast first pass with mini, then rescore top_m with a stronger model."""
#     first = score_pairs(papers_df, patents_df, intent=intent, model=fast_model, concurrency=concurrency)
#     top = first.head(max(1, int(top_m))).copy()
#     # Rescore top_m rows using strong model (sync)
#     rescored_rows = []
#     for _, r in top.iterrows():
#         p = pd.Series({
#             "doi": r.get("paper_doi",""), "title": r.get("paper_title",""),
#             "first_author": r.get("paper_first_author",""), "last_author": r.get("paper_last_author",""),
#             "citations_total": r.get("paper_citations",""), "abstract": ""
#         })
#         t = pd.Series({
#             "publication_number": r.get("patent_id",""), "title": r.get("patent_title",""),
#             "lead_inventor": r.get("patent_inventor",""), "abstract": ""
#         })
#         prompt, ov = _make_pair_prompt(p, t, intent)
#         obj = _call_gpt_json(prompt, model=strong_model)
#         r["fit_score"] = _safe_int(obj.get("score"))
#         r["summary"] = obj.get("summary","")
#         rescored_rows.append(r)

#     rescored = pd.DataFrame(rescored_rows)
#     first.update(rescored.set_index(first.head(len(rescored)).index))
#     return first.sort_values("fit_score", ascending=False, na_position="last")

# # =========================
# # High-level pipeline
# # =========================
# def run_gpt_pipeline(
#     papers_df: pd.DataFrame,
#     patents_df: pd.DataFrame,
#     intent: str = "Find market fit and propose a founding team",
#     topk: int = 10,
#     model: str = FAST_MODEL,
#     use_two_stage: bool = True,
#     prefilter: bool = True,
#     top_papers_prefilter: int = 10,
#     top_patents_prefilter: int = 10,
#     concurrency: int = 6,
# ) -> Tuple[pd.DataFrame, str]:
#     """
#     End-to-end:
#       1) (optional) prefilter to reduce pairs
#       2) score all remaining pairs with async concurrency (or two-stage)
#       3) select top-K
#       4) summarize top-K
#       5) render HTML
#     Returns (top_df, html_str).
#     """
#     if prefilter:
#         papers_df, patents_df = prefilter_pairs(papers_df, patents_df,
#                                                 top_papers=top_papers_prefilter,
#                                                 top_patents=top_patents_prefilter)

#     if use_two_stage:
#         scored = score_pairs_two_stage(papers_df, patents_df, intent=intent,
#                                        fast_model=FAST_MODEL, strong_model=FAST_MODEL, top_m=max(10, topk),
#                                        concurrency=concurrency)
#     else:
#         scored = score_pairs(papers_df, patents_df, intent=intent, model=model, concurrency=concurrency)

#     top_df = scored.head(max(1, int(topk)))
#     summary = summarize_top_pairs(top_df, intent=intent, model=model)
#     html_str = render_matches_html(top_df, exec_summary=summary)
#     return top_df, html_str

# paper_patent_fit_module.py
from __future__ import annotations
import os, re, json, html, hashlib, asyncio
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd

# =========================
# Config / Constants
# =========================
FAST_MODEL = "gpt-4o-mini"   # the ONLY model used
CACHE_DIR = Path(".gpt_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Token trimming (speed win)
MAX_TITLE_CH = 220
MAX_ABS_CH   = 1200

# =========================
# API key / client handling
# =========================
def load_api_key(path: str | Path = "secrets/openai_key.txt") -> None:
    """
    Ensure OPENAI_API_KEY is set. If missing, read a one-line key from `path`.
    No-op if already set.
    """
    if os.getenv("OPENAI_API_KEY"):
        return
    p = Path(path)
    if p.exists():
        key = p.read_text(encoding="utf-8").strip()
        if key:
            os.environ["OPENAI_API_KEY"] = key

def _get_client():
    """Return (client, FAST_MODEL). If import/env fails, returns (None, None)."""
    try:
        from openai import OpenAI
        if not os.getenv("OPENAI_API_KEY"):
            load_api_key()
        return OpenAI(), FAST_MODEL
    except Exception:
        return None, None

def _get_async_client():
    """Return (async_client, FAST_MODEL)."""
    try:
        from openai import AsyncOpenAI
        if not os.getenv("OPENAI_API_KEY"):
            load_api_key()
        return AsyncOpenAI(), FAST_MODEL
    except Exception:
        return None, None

# =========================
# Small helpers
# =========================
def _norm(s: str) -> str:
    return (s or "").strip()

def _split_names(s: str) -> List[str]:
    if not isinstance(s, str): return []
    parts = re.split(r"[;,]", s)
    return [p.strip() for p in parts if p.strip()]

def _overlap_names(paper_first: str, paper_last: str, inventors: List[str]) -> Dict[str, Any]:
    authors = [a for a in [_norm(paper_first), _norm(paper_last)] if a]
    al, il = {a.lower() for a in authors}, {i.lower() for i in inventors}
    overlap = sorted(al.intersection(il))
    return {"overlap": bool(overlap), "overlap_names": overlap, "authors": authors, "inventors": inventors}

def _safe_int(x, lo=1, hi=10) -> int:
    try:
        v = int(x); return max(lo, min(hi, v))
    except Exception:
        return 0

def _clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return (s[:n] + "…") if len(s) > n else s

# =========================
# Caching (prompt → JSON)
# =========================
def _prompt_key(prompt: str) -> Path:
    h = hashlib.sha256((FAST_MODEL + "\n" + prompt).encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{h}.json"

def _read_cache(prompt: str) -> Optional[dict]:
    p = _prompt_key(prompt)
    if p.exists():
        try:
            return json.loads(p.read_text("utf-8"))
        except Exception:
            return None
    return None

def _write_cache(prompt: str, obj: dict) -> None:
    _prompt_key(prompt).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")

# =========================
# Prompting & calls (FAST model only)
# =========================
def _make_pair_prompt(paper: pd.Series, patent: pd.Series, intent: str) -> Tuple[str, Dict[str, Any]]:
    first_author = _norm(paper.get("first_author", ""))
    last_author  = _norm(paper.get("last_author",  ""))
    inventors    = _split_names(_norm(patent.get("lead_inventor", "")))

    ov   = _overlap_names(first_author, last_author, inventors)
    doi  = _norm(paper.get("doi", ""))
    pdf  = _norm(paper.get("pdf_url", ""))
    pubn = _norm(patent.get("publication_number", ""))

    paper_title   = _clip(paper.get('title',''), MAX_TITLE_CH)
    paper_abs     = _clip(paper.get('abstract',''), MAX_ABS_CH)
    patent_title  = _clip(patent.get('title',''), MAX_TITLE_CH)
    patent_abs    = _clip(patent.get('abstract',''), MAX_ABS_CH)

    prompt = f"""
You are a senior venture & IP analyst. Goal: **{intent}**.

Compare the RESEARCH PAPER and the PATENT and output strict JSON with keys:
- score (int 1..10)
- summary (string, 1–3 sentences)
- ranked_authors: array of {{name, score_1to10, reason}} sorted best-first
- ranked_inventors: array of {{name, score_1to10, reason}} sorted best-first
- overlap_bonus_applied (boolean)

If any paper author also appears among inventors, apply a positive but defensible bias.

RESEARCH PAPER
- DOI: {doi}
- Title: {paper_title}
- Authors: {first_author} ; {last_author}
- Citations: {paper.get('citations_total','')}
- Abstract: {paper_abs}
- PDF: {pdf}

PATENT
- Publication #: {pubn}
- Title: {patent_title}
- Lead Inventor(s): {', '.join(inventors) if inventors else '(none)'}
- Abstract: {patent_abs}

Name-overlap hint (precomputed):
- overlap: {ov['overlap']}
- overlap_names: {ov['overlap_names']}
"""
    return prompt.strip(), ov

def _call_gpt_json(prompt: str) -> Dict[str, Any]:
    """
    Structured JSON via Chat Completions JSON mode (sync).
    FAST model only. Uses disk cache; falls back to mock if client unavailable.
    """
    cached = _read_cache(prompt)
    if cached:
        return cached

    client, model = _get_client()
    if client is None or model is None:
        return {"score": 5, "summary": "(mock) average fit.",
                "ranked_authors": [], "ranked_inventors": [], "overlap_bonus_applied": False}

    system = ("You are precise. Respond with strict JSON only containing: "
              "score, summary, ranked_authors, ranked_inventors, overlap_bonus_applied.")
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=400,  # gpt-4o-mini accepts max_tokens
    )
    raw = resp.choices[0].message.content.strip()
    try:
        obj = json.loads(raw)
    except Exception:
        mobj = re.search(r"\{.*\}\s*$", raw, flags=re.DOTALL)
        obj = json.loads(mobj.group(0)) if mobj else {
            "score": 0, "summary": raw, "ranked_authors": [], "ranked_inventors": [], "overlap_bonus_applied": False
        }
    _write_cache(prompt, obj)
    return obj

async def _acall_gpt_json(prompt: str) -> Dict[str, Any]:
    cached = _read_cache(prompt)
    if cached:
        return cached

    aclient, model = _get_async_client()
    if aclient is None or model is None:
        return {"score": 5, "summary": "(mock) average fit.",
                "ranked_authors": [], "ranked_inventors": [], "overlap_bonus_applied": False}

    system = ("You are precise. Respond with strict JSON only containing: "
              "score, summary, ranked_authors, ranked_inventors, overlap_bonus_applied.")
    resp = await aclient.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[{"role":"system","content":system},
                  {"role":"user","content":prompt}],
        temperature=0,
        max_tokens=400,
    )
    raw = resp.choices[0].message.content.strip()
    try:
        obj = json.loads(raw)
    except Exception:
        mobj = re.search(r"\{.*\}\s*$", raw, flags=re.DOTALL)
        obj = json.loads(mobj.group(0)) if mobj else {
            "score": 0, "summary": raw, "ranked_authors": [], "ranked_inventors": [], "overlap_bonus_applied": False
        }
    _write_cache(prompt, obj)
    return obj

# =========================
# Prefilter (cheap pruning)
# =========================
def prefilter_pairs(papers_df: pd.DataFrame, patents_df: pd.DataFrame,
                    top_papers: int = 15, top_patents: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reduce cartesian product size quickly (adjust thresholds to taste)."""
    if "citations_total" in papers_df.columns:
        papers_df = papers_df.sort_values("citations_total", ascending=False).head(top_papers)
    else:
        papers_df = papers_df.head(top_papers)
    patents_df = patents_df.head(top_patents)
    return papers_df, patents_df

# =========================
# Scoring (async + sync wrapper) — FAST model only
# =========================
async def score_pairs_async(
    papers_df: pd.DataFrame,
    patents_df: pd.DataFrame,
    intent: str = "Find market fit and propose a founding team",
    concurrency: int = 6,
) -> pd.DataFrame:
    sem = asyncio.Semaphore(concurrency)
    tasks: List[asyncio.Task] = []

    async def worker(p: pd.Series, t: pd.Series):
        async with sem:
            prompt, ov = _make_pair_prompt(p, t, intent)
            obj = await _acall_gpt_json(prompt)
            ra = obj.get("ranked_authors", []) or []
            ri = obj.get("ranked_inventors", []) or []
            return {
                "paper_doi": p.get("doi",""),
                "paper_title": p.get("title",""),
                "paper_first_author": p.get("first_author",""),
                "paper_last_author": p.get("last_author",""),
                "paper_citations": p.get("citations_total",""),
                "patent_id": t.get("publication_number",""),
                "patent_title": t.get("title",""),
                "patent_inventor": t.get("lead_inventor",""),
                "fit_score": _safe_int(obj.get("score")),
                "summary": obj.get("summary",""),
                "top_author": (ra[0]["name"] if ra else ""),
                "top_author_fit": (ra[0].get("score_1to10") if ra else ""),
                "top_inventor": (ri[0]["name"] if ri else ""),
                "top_inventor_fit": (ri[0].get("score_1to10") if ri else ""),
                "overlap_bonus_applied": bool(obj.get("overlap_bonus_applied", False) or ov["overlap"]),
                "ranked_authors_json": json.dumps(ra, ensure_ascii=False),
                "ranked_inventors_json": json.dumps(ri, ensure_ascii=False),
            }

    for _, p in papers_df.iterrows():
        for _, t in patents_df.iterrows():
            tasks.append(asyncio.create_task(worker(p, t)))

    rows = await asyncio.gather(*tasks)
    df = pd.DataFrame(rows)
    df["fit_score"] = pd.to_numeric(df["fit_score"], errors="coerce")
    return df.sort_values("fit_score", ascending=False, na_position="last")

def score_pairs(
    papers_df: pd.DataFrame,
    patents_df: pd.DataFrame,
    intent: str = "Find market fit and propose a founding team",
    concurrency: int = 6,
) -> pd.DataFrame:
    """Sync facade; runs the async scorer with concurrency (FAST model only)."""
    try:
        return asyncio.run(score_pairs_async(papers_df, patents_df, intent, concurrency))
    except RuntimeError:
        # Already in an event loop (e.g., Jupyter/ASGI). Use loop directly.
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(score_pairs_async(papers_df, patents_df, intent, concurrency))

# =========================
# Executive summary (FAST model only)
# =========================
def summarize_top_pairs(
    top_df: pd.DataFrame,
    intent: str = "Find market fit and propose a founding team",
) -> Dict[str, Any]:
    client, model = _get_client()
    if client is None or model is None:
        recs = []
        for i, r in enumerate(top_df.itertuples(index=False), 1):
            recs.append({"rank": i, "paper_title": getattr(r,"paper_title",""),
                         "patent_title": getattr(r,"patent_title",""),
                         "reason": f"Fit {getattr(r,'fit_score','')}; overlap={getattr(r,'overlap_bonus_applied',False)}."})
        return {"summary": f"Top {len(recs)} pairs by fit for: {intent}.", "recommendations": recs}

    items = []
    for i, r in enumerate(top_df.itertuples(index=False), 1):
        items.append({
            "rank": i,
            "paper_title": getattr(r,"paper_title",""),
            "paper_doi": getattr(r,"paper_doi",""),
            "patent_title": getattr(r,"patent_title",""),
            "patent_id": getattr(r,"patent_id",""),
            "fit_score": int(getattr(r,"fit_score",0) or 0),
            "overlap_bonus": bool(getattr(r,"overlap_bonus_applied",False)),
            "top_author": getattr(r,"top_author",""),
            "top_inventor": getattr(r,"top_inventor",""),
        })

    system = ("You are a concise venture/IP analyst. Write strict JSON with keys: "
              "'summary' (3–5 sentences) and 'recommendations' "
              "(array of {rank,paper_title,patent_title,reason}, 1–2 lines each).")
    user = {"intent_task": intent, "top_pairs": items}

    resp = client.chat.completions.create(
        model=model,
        response_format={"type":"json_object"},
        messages=[{"role":"system","content":system},
                  {"role":"user","content":json.dumps(user, ensure_ascii=False)}],
        temperature=0,
        max_tokens=400   # mini accepts max_tokens
    )
    raw = resp.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"summary": f"Top {len(items)} by fit.",
                "recommendations": [
                    {"rank": it["rank"], "paper_title": it["paper_title"], "patent_title": it["patent_title"],
                     "reason": f"Fit {it['fit_score']}; overlap={it['overlap_bonus']}."} for it in items]}

# =========================
# HTML rendering
# =========================
def _linkify(url: str, text: str) -> str:
    t = html.escape(text or "link")
    u = (url or "").strip()
    return f'<a href="{html.escape(u, quote=True)}" target="_blank" rel="noopener noreferrer">{t}</a>' if u else t

def _details(text: str) -> str:
    if not isinstance(text, str) or not text.strip(): return ""
    short = html.escape(text[:180]) + ("..." if len(text) > 180 else "")
    full  = html.escape(text)
    return f'<details><summary>{short}</summary><div style="white-space:pre-wrap;margin-top:.5rem;">{full}</div></details>'

def _pre(json_text: str) -> str:
    if not json_text: return ""
    return f'<details><summary>show</summary><pre style="white-space:pre-wrap;margin:.5rem 0;">{html.escape(json_text)}</pre></details>'

def render_matches_html(
    df: pd.DataFrame,
    exec_summary: Optional[Dict[str, Any]] = None,
    table_id: str = "matches_table"
) -> str:
    dfx = df.copy()
    dfx["fit_score"] = pd.to_numeric(dfx["fit_score"], errors="coerce")
    anchors = [f"row-{i}" for i in range(1, len(dfx)+1)]

    table = pd.DataFrame({
        "": [f'<a id="{aid}"></a>' for aid in anchors],
        "#": range(1, len(dfx)+1),
        "Paper": [_linkify(dfx.get("paper_doi","").iloc[i], dfx.get("paper_title","").iloc[i]) for i in range(len(dfx))],
        "First Author": dfx.get("paper_first_author",""),
        "Last Author": dfx.get("paper_last_author",""),
        "Citations": dfx.get("paper_citations",""),
        "Patent ID": dfx.get("patent_id",""),
        "Patent Title": dfx.get("patent_title",""),
        "Lead Inventor": dfx.get("patent_inventor",""),
        "Fit Score": dfx["fit_score"].map(lambda x: f"{x:.0f}" if pd.notna(x) else ""),
        "Summary": dfx.get("summary","").map(_details),
        "Top Author (fit)": dfx.apply(lambda r: f"{r.get('top_author','')} ({r.get('top_author_fit','')})", axis=1),
        "Top Inventor (fit)": dfx.apply(lambda r: f"{r.get('top_inventor','')} ({r.get('top_inventor_fit','')})", axis=1),
        "Overlap Bonus?": dfx.get("overlap_bonus_applied", False).map(lambda x: "Yes" if x else "No"),
        "Ranked Authors": dfx.get("ranked_authors_json","").map(_pre),
        "Ranked Inventors": dfx.get("ranked_inventors_json","").map(_pre),
    })
    table_html = table.to_html(index=False, escape=False, border=0, table_id=table_id, classes="display nowrap compact")

    summary_html = ""
    if exec_summary:
        items = ""
        for rec in exec_summary.get("recommendations", []):
            rid = int(rec.get("rank", 0))
            items += (
                f'<li><a href="#row-{rid}"><strong>#{rid}</strong> '
                f'{html.escape(rec.get("paper_title",""))} ↔ {html.escape(rec.get("patent_title",""))}</a>'
                f'<br><span style="color:#555;">{html.escape(rec.get("reason",""))}</span></li>'
            )
        summary_html = f"""
        <div class="card" style="margin-bottom:1rem;">
          <div class="card-body">
            <h3 class="card-title">Executive summary</h3>
            <p>{html.escape(exec_summary.get("summary",""))}</p>
            <h5>Top recommendations</h5>
            <ol>{items}</ol>
          </div>
        </div>
        """

    cols_json = json.dumps(list(table.columns))
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Paper–Patent Matches (Top-K)</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"/>
  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css"/>
  <style>
    body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:1.5rem;background:#fafafa}}
    table.dataTable thead th{{white-space:nowrap}} details>summary{{cursor:pointer;color:#0d6efd}}
    details[open]>summary{{color:#0a58ca}} .card{{box-shadow:0 2px 10px rgba(0,0,0,.05)}}
  </style>
</head>
<body>
  {summary_html}
  <h2>Top Matches (Paper ↔ Patent) with Author/Inventor Ranking</h2>
  {table_html}
  <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
  <script>
    $(function(){{
      const names={cols_json}; const scoreIdx=names.indexOf("Fit Score");
      $('#{table_id}').DataTable({{paging:true,searching:true,order:scoreIdx>=0?[[scoreIdx,'desc']]:[[0,'asc']],scrollX:true,responsive:true,pageLength:10,lengthMenu:[10,25,50,100]}});
    }});
  </script>
</body>
</html>"""

# =========================
# High-level pipeline (FAST model only)
# =========================
def run_gpt_pipeline(
    papers_df: pd.DataFrame,
    patents_df: pd.DataFrame,
    intent: str = "Find market fit and propose a founding team",
    topk: int = 10,
    prefilter: bool = True,
    top_papers_prefilter: int = 15,
    top_patents_prefilter: int = 15,
    concurrency: int = 8,
) -> Tuple[pd.DataFrame, str]:
    """
    End-to-end (FAST model only):
      1) optional prefilter to reduce pairs
      2) score all remaining pairs with async concurrency
      3) select top-K
      4) summarize top-K
      5) render HTML
    Returns (top_df, html_str).
    """
    if prefilter:
        papers_df, patents_df = prefilter_pairs(
            papers_df, patents_df,
            top_papers=top_papers_prefilter,
            top_patents=top_patents_prefilter
        )

    scored = score_pairs(papers_df, patents_df, intent=intent, concurrency=concurrency)
    top_df = scored.head(max(1, int(topk)))
    summary = summarize_top_pairs(top_df, intent=intent)
    html_str = render_matches_html(top_df, exec_summary=summary)
    return top_df, html_str
