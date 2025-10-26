import os
import asyncio
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response, url_for, current_app, stream_with_context

from paper_search_module import (
    find_and_extract,
    render_paper_search_output,
    build_author_conference_network_async,
    plot_author_conference_graph
)
from patent_search_module import (
    search_patents_df,
    render_patent_search_output
)
# Pull in internals so we can stream true per-pair progress
from paper_patent_fit_module import (
    run_gpt_pipeline,              # still available if you want to call it directly
    prefilter_pairs,
    render_matches_html,
    summarize_top_pairs,
    _make_pair_prompt,
    _acall_gpt_json,
    _safe_int,
)

app = Flask(__name__)

# ------------------------------
# Key loader (unchanged)
# ------------------------------
from pathlib import Path
KEY_ENV = "OPENAI_API_KEY"
KEY_FILE = Path("secrets/openai_key.txt")

def load_openai_key() -> str:
    key = os.getenv(KEY_ENV, "").strip()
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8").strip()
    if not key or not key.startswith("sk-") or len(key) < 20:
        raise RuntimeError(
            "OPENAI_API_KEY missing/invalid. Set env OPENAI_API_KEY or place a valid key in secrets/openai_key.txt"
        )
    if " " in key or "\n" in key or "…" in key or key.endswith(("=", "==")):
        raise RuntimeError("OPENAI_API_KEY looks malformed (contains spaces/newlines/ellipsis). Re-copy it.")
    print("Using OPENAI_API_KEY:", key[:8] + "…" + key[-6:])
    os.environ[KEY_ENV] = key
    return key

load_openai_key()


# ------------------------------
# Helpers
# ------------------------------
def _sse(obj: dict) -> str:
    """Format dict as an SSE data line."""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

def _parse_count(req, name, default=10, min_val=1, max_val=100):
    try:
        v = int(req.get(name, default))
    except (TypeError, ValueError):
        v = default
    v = max(min_val, v)
    if max_val is not None:
        v = min(max_val, v)
    return v


# ------------------------------
# Home page
# ------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index_alt.html")  # shows the form


# ------------------------------
# Search: return a page that starts SSE
# ------------------------------
@app.route("/search", methods=["POST"])
def search():
    # Read inputs
    paper_query  = (request.form.get("paper_query")  or "").strip()
    patent_query = (request.form.get("patent_query") or "").strip()
    intent       = (request.form.get("intent")       or "").strip()
    mode         = request.form.get("mode", "notext")
    paper_n      = _parse_count(request.form, "paper_n", 10, 1, 100)
    patent_n     = _parse_count(request.form, "patent_n", 10, 1, 100)

    # Require intent
    if not intent:
        return render_template(
            "index_alt.html",
            warning="Please provide an Intent before searching.",
            paper_query=paper_query, patent_query=patent_query,
            mode=mode, paper_n=paper_n, patent_n=patent_n, intent=intent
        ), 400

    if not paper_query and not patent_query:
        return render_template(
            "index_alt.html",
            warning="Please enter a Paper or Patent search term (or both).",
            paper_query=paper_query, patent_query=patent_query,
            mode=mode, paper_n=paper_n, patent_n=patent_n, intent=intent
        ), 400

    # Render a results shell that opens the SSE stream (progress bar + live status + results slot)
    # We pass params back to JS so it can build `/analyze_sse?...`
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <title>Search Results</title>
      <meta name="viewport" content="width=device-width, initial-scale=1"/>
      <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
      <style>
        body {{ background:#f9f9f9; }}
      </style>
    </head>
    <body class="container mt-5">
      <h1 class="mb-3">Search & Analysis</h1>
      <p class="text-muted mb-4">
        <strong>Intent:</strong> {intent}<br/>
        <strong>Paper mode:</strong> {mode} |
        <strong>Papers:</strong> {paper_n} |
        <strong>Patents:</strong> {patent_n}
      </p>

      <!-- Progress UI -->
      <div id="progress-wrap" class="mt-2">
        <div class="progress" style="height: 24px;">
          <div id="progress-bar" class="progress-bar progress-bar-striped progress-bar-animated"
               role="progressbar" style="width: 0%;">0%</div>
        </div>
        <div id="progress-status" class="mt-2 small text-muted">Starting…</div>
      </div>

      <!-- Live sections -->
      <div id="papers" class="mt-4"></div>
      <div id="patents" class="mt-4"></div>

      <!-- GPT final HTML report goes here -->
      <div id="gpt-results" class="mt-5"></div>

      <a href="/" class="btn btn-secondary mt-4 mb-5">← Back</a>

      <script>
        (function(){{
          const params = new URLSearchParams({{
            paper_query: {json.dumps(paper_query)},
            patent_query: {json.dumps(patent_query)},
            intent: {json.dumps(intent)},
            mode: {json.dumps(mode)},
            paper_n: {paper_n},
            patent_n: {patent_n}
          }});

          const wrap   = document.getElementById('progress-wrap');
          const bar    = document.getElementById('progress-bar');
          const status = document.getElementById('progress-status');
          const papersDiv  = document.getElementById('papers');
          const patentsDiv = document.getElementById('patents');
          const gptDiv     = document.getElementById('gpt-results');

          function setPct(pct) {{
            const p = Math.max(0, Math.min(100, Math.floor(pct)));
            bar.style.width = p + '%';
            bar.textContent = p + '%';
          }}

          const es = new EventSource('/analyze_sse?' + params.toString());

          es.onmessage = (evt) => {{
            try {{
              const msg = JSON.parse(evt.data);
              if (msg.type === 'progress') {{
                setPct(msg.pct || 0);
                if (msg.note) status.textContent = msg.note;
                if (msg.phase === 'papers' && msg.html) papersDiv.innerHTML = msg.html;
                if (msg.phase === 'patents' && msg.html) patentsDiv.innerHTML = msg.html;
              }} else if (msg.type === 'final') {{
                gptDiv.innerHTML = msg.html || '<p>No output.</p>';
                status.textContent = 'Done.';
                bar.classList.remove('progress-bar-animated');
                es.close();
              }} else if (msg.type === 'error') {{
                status.textContent = 'Error: ' + (msg.error || 'unknown');
                bar.classList.remove('progress-bar-animated', 'progress-bar-striped');
                bar.classList.add('bg-danger');
                es.close();
              }}
            }} catch (e) {{
              console.error('SSE parse error', e);
            }}
          }};

          es.onerror = () => {{
            status.textContent = 'Connection lost.';
            bar.classList.remove('progress-bar-animated', 'progress-bar-striped');
            bar.classList.add('bg-danger');
            es.close();
          }};
        }})();
      </script>
    </body>
    </html>
    """


# ------------------------------
# SSE endpoint: streams progress + final HTML
# ------------------------------
@app.route("/analyze_sse", methods=["GET"])
def analyze_sse():
    # Pull query params (SSE uses GET)
    paper_query  = (request.args.get("paper_query")  or "").strip()
    patent_query = (request.args.get("patent_query") or "").strip()
    intent       = (request.args.get("intent")       or "").strip()
    mode         = request.args.get("mode", "notext")
    paper_n      = _parse_count(request.args, "paper_n", 10, 1, 100)
    patent_n     = _parse_count(request.args, "patent_n", 10, 1, 100)

    if not intent:
        return Response(_sse({"type":"error","error":"Missing intent"}), mimetype="text/event-stream")

    @stream_with_context
    def generate():
        # Phase 1: fetch papers
        try:
            yield _sse({"type":"progress","pct": 2, "phase":"papers","note":"Fetching papers…"})
            paper_df = find_and_extract(paper_query, n=paper_n, mode=mode, print_output=False)
            paper_html = render_paper_search_output(paper_df)
            yield _sse({"type":"progress","pct": 18, "phase":"papers","note":"Papers ready","html": paper_html})
        except Exception as e:
            yield _sse({"type":"error","error": f"Paper fetch failed: {e}"})
            return

        # Phase 2: fetch patents
        try:
            yield _sse({"type":"progress","pct": 20, "phase":"patents","note":"Fetching patents…"})
            patent_df = search_patents_df(patent_query, n_results=patent_n)
            patent_html = render_patent_search_output(patent_df)
            yield _sse({"type":"progress","pct": 36, "phase":"patents","note":"Patents ready","html": patent_html})
        except Exception as e:
            yield _sse({"type":"error","error": f"Patent fetch failed: {e}"})
            return

        # Optional prefilter (keeps UI snappy on large inputs)
        paper_df, patent_df = prefilter_pairs(
            paper_df, patent_df,
            top_papers=min(15, len(paper_df)),
            top_patents=min(15, len(patent_df))
        )

        # Phase 3: per-pair GPT scoring with live progress
        total = max(1, len(paper_df) * len(patent_df))
        done = 0

        async def score_all():
            sem = asyncio.Semaphore(6)
            tasks = []

            async def worker(p_row, t_row):
                nonlocal done
                async with sem:
                    prompt, ov = _make_pair_prompt(p_row, t_row, intent)
                    obj = await _acall_gpt_json(prompt)
                    ra = obj.get("ranked_authors", []) or []
                    ri = obj.get("ranked_inventors", []) or []
                    res = {
                        "paper_doi": p_row.get("doi",""),
                        "paper_title": p_row.get("title",""),
                        "paper_first_author": p_row.get("first_author",""),
                        "paper_last_author": p_row.get("last_author",""),
                        "paper_citations": p_row.get("citations_total",""),
                        "patent_id": t_row.get("publication_number",""),
                        "patent_title": t_row.get("title",""),
                        "patent_inventor": t_row.get("lead_inventor",""),
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
                    done += 1
                    pct = 36 + (done / total) * 60  # 36→96% during scoring
                    # Yield a server push *from the async loop* by placing onto a queue
                    return res, pct

            for _, p in paper_df.iterrows():
                for _, t in patent_df.iterrows():
                    tasks.append(asyncio.create_task(worker(p, t)))

            rows = []
            for coro in asyncio.as_completed(tasks):
                res, pct = await coro
                rows.append(res)
                # Push progress out to the generator synchronously via a future
                queue.append({"type":"progress","pct": pct, "note": f"Scored {done}/{total} pairs"})
            return rows

        # A tiny bridge: async→sync messages via a list we drain in the generator
        queue: list = []

        # Run the async scoring in a private loop
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            rows = loop.run_until_complete(score_all())
        except Exception as e:
            yield _sse({"type":"error","error": f"Scoring failed: {e}"})
            return
        finally:
            # Drain any remaining progress items
            while queue:
                yield _sse(queue.pop(0))
            try:
                loop.close()
            except Exception:
                pass

        # Emit any last progress if needed
        yield _sse({"type":"progress","pct": 98, "note":"Finalizing report…"})

        # Build final HTML
        df = pd.DataFrame(rows)
        if not df.empty:
            df["fit_score"] = pd.to_numeric(df["fit_score"], errors="coerce")
            df = df.sort_values("fit_score", ascending=False, na_position="last")
            top_df = df.head(10)
            summary = summarize_top_pairs(top_df, intent=intent)
            html_report = render_matches_html(top_df, exec_summary=summary)
        else:
            html_report = "<p>No results.</p>"

        yield _sse({"type":"final", "html": html_report})

    return Response(generate(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002, debug=True)