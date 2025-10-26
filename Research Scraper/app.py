import os
import asyncio
from flask import Flask, render_template, request, jsonify, Response, url_for, current_app

from paper_search_module import (
    find_and_extract,
    render_paper_search_output,
    build_author_conference_network_async,  # correct name
    plot_author_conference_graph              # updated plotting function
)
from patent_search_module import (
    search_patents_df,
    render_patent_search_output
)
from paper_patent_fit_module import (
    run_gpt_pipeline
)

app = Flask(__name__)

# key_loader.py
import os, re
from pathlib import Path

KEY_ENV = "OPENAI_API_KEY"
KEY_FILE = Path("/Users/tyrellto/Documents/hack2025/Research Scraper/secrets/openai_key.txt")

def load_openai_key() -> str:
    """
    Load key from env if present, else from secrets file.
    Validate, strip whitespace/newlines, and hard-fail if suspicious.
    """
    key = os.getenv(KEY_ENV, "").strip()
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8").strip()

    # Basic validation (don’t print the whole key!)
    if not key or not key.startswith("sk-") or len(key) < 20:
        raise RuntimeError(
            "OPENAI_API_KEY missing/invalid. Set env OPENAI_API_KEY or place a valid key in secrets/openai_key.txt"
        )
    # common copy-paste mistakes
    if " " in key or "\n" in key or "…" in key or key.endswith(("=", "==")):
        raise RuntimeError("OPENAI_API_KEY looks malformed (contains spaces/newlines/ellipsis). Re-copy it.")

    # mask in logs
    print("Using OPENAI_API_KEY:", key[:8] + "…" + key[-6:])
    os.environ[KEY_ENV] = key   # ensure the SDK sees it
    return key

load_openai_key()
# ------------------------------
# Home page with search form
# ------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")  # Form posts to /search

# ------------------------------
# Search endpoint
# ------------------------------
@app.route("/search", methods=["POST"])
def search():
    # Read inputs
    paper_query  = (request.form.get("paper_query")  or "").strip()
    patent_query = (request.form.get("patent_query") or "").strip()
    intent       = (request.form.get("intent")       or "").strip()
    mode         = request.form.get("mode", "notext")

    # Parse counts (keep within sane bounds)
    def _parse_count(name, default=10, min_val=1, max_val=100):
        try:
            v = int(request.form.get(name, default))
        except (TypeError, ValueError):
            v = default
        v = max(min_val, v)
        if max_val is not None:
            v = min(max_val, v)
        return v

    paper_n  = _parse_count("paper_n", 10, 1, 100)
    patent_n = _parse_count("patent_n", 10, 1, 100)

    # ✅ Require intent; if missing, go back to index with a warning and DO NOT proceed
    if not intent:
        return render_template(
            "index.html",
            warning="Please provide an Intent before searching.",
            # Prefill values so the user doesn’t lose what they typed
            paper_query=paper_query,
            patent_query=patent_query,
            mode=mode,
            paper_n=paper_n,
            patent_n=patent_n,
            intent=intent,
        ), 400

    # If you also want to require at least one of the queries, uncomment below:
    if not paper_query and not patent_query:
        return render_template(
            "index.html",
            warning="Please enter a Paper or Patent search term (or both).",
            paper_query=paper_query, patent_query=patent_query,
            mode=mode, paper_n=paper_n, patent_n=patent_n, intent=intent
        ), 400

    # Fetch papers
    try:
        paper_df = find_and_extract(paper_query, n=paper_n, mode=mode, print_output=False)
    except Exception as e:
        return f"Error while fetching papers: {e}", 500

    paper_html_table = render_paper_search_output(paper_df)

    try:
        patent_df = search_patents_df(patent_query, n_results=patent_n)
    except Exception as e:
        return f"Error while fetching patents: {e}", 500

    patent_html_table = render_patent_search_output(patent_df)

    top_df, gpt_html_table = run_gpt_pipeline(paper_df, patent_df, intent=intent, topk=10)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Search Results</title>
        <link rel="stylesheet"
            href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
        <style>
            body {{
                background-color: #f9f9f9;
            }}
            #spinner {{
                display:none;
                width: 3rem;
                height: 3rem;
                border: 0.4em solid #f3f3f3;
                border-top: 0.4em solid #007bff;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 20px auto;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            iframe {{
                background-color: #fff;
            }}
        </style>
    </head>
    <body class="container mt-5">
        <h1>Search Results for '{paper_query.strip()}'</h1>
        <p>Mode: <strong>{mode}</strong> | Number of results: <strong>{paper_n}</strong></p>

        <div class="mt-4">{paper_html_table}</div>

        <h1>Search Results for '{patent_query.strip()}'</h1>
        <p>Number of results: <strong>{patent_n}</strong></p>

        <div class="mt-4">{patent_html_table}</div>

        <!-- GPT pair scoring + summary -->
        <hr class="my-5">
        <div class="card">
        <div class="card-body">
            <h2 class="card-title mb-3">GPT Matches & Executive Summary</h2>
            <p class="text-muted mb-3">
            <strong>Intent:</strong> {intent}
            </p>
            <!-- gpt_html_table is already full HTML; insert directly -->
            <div class="mt-3">{gpt_html_table}</div>
        </div>
        </div>

        <div class="mt-4">
            <label for="min_shared_conferences">Min Shared Conferences:</label>
            <input type="number" id="min_shared_conferences" value="0" min="0" class="form-control" style="width: auto;">
        </div>
        <div class="mt-4">
            <label for="min_citations">Min Citations:</label>
            <input type="number" id="min_citations" value="0" min="0" class="form-control" style="width: auto;">
        </div>

        <div class="form-check form-switch mt-4">
            <input class="form-check-input" type="checkbox" id="toggle_network">
            <label class="form-check-label" for="toggle_network">Show Author Conference Network</label>
        </div>

        <div id="spinner"></div>
        <div id="graph_container"></div>

        <a href="/" class="btn btn-secondary mt-4 mb-5">← Back</a>

        <script>
            const toggle = document.getElementById('toggle_network');
            const spinner = document.getElementById('spinner');
            const container = document.getElementById('graph_container');
            const minSharedConferencesInput = document.getElementById('min_shared_conferences');
            const minCitationsInput = document.getElementById('min_citations');

            toggle.addEventListener('change', async () => {{
                if (toggle.checked) {{
                    spinner.style.display = 'block';
                    container.innerHTML = '';
                    try {{
                        const formData = new FormData();
                        formData.append('paper_query', '{paper_query}');
                        formData.append('patent_query', '{patent_query}');
                        formData.append('mode', '{mode}');
                        formData.append('paper_n', '{paper_n}');
                        formData.append('patent_n', '{patent_n}');
                        formData.append('min_shared_conferences', minSharedConferencesInput.value);
                        formData.append('min_citations', minCitationsInput.value);
                        
                        const resp = await fetch('/network', {{
                            method: 'POST',
                            body: formData
                        }});
                        const data = await resp.json();
                        if (data.success) {{
                            container.innerHTML = data.html;
                        }} else {{
                            container.innerHTML = '<p style="color:red;">Error: ' + data.error + '</p>';
                        }}
                    }} catch(err) {{
                        container.innerHTML = '<p style="color:red;">' + err + '</p>';
                    }} finally {{
                        spinner.style.display = 'none';
                    }}
                }} else {{
                    container.innerHTML = '';
                }}
            }});
        </script>
    </body>
    </html>
    """



# ------------------------------
# Network endpoint for toggle
# ------------------------------
@app.route("/network", methods=["POST"])
async def network():
    paper_query = request.form.get("paper_query")
    # patent_query = request.form.get("patent_query")
    mode = request.form.get("mode", "sections")
    try:
        n = int(request.form.get("n", 10))
        n = max(1, n)
    except ValueError:
        n = 10
        
    # Get the filtering values from the form
    min_shared_conferences = int(request.form.get("min_shared_conferences", 2))  # Default to 2
    min_citations = int(request.form.get("min_citations", 20))  # Default to 20
    
    try:
        df = find_and_extract(paper_query, n=n, mode=mode, print_output=False)
        
        # Async function to build and plot the graph, consuming the async generator properly
        async def generate_graph():
            result = []
            async for G in build_author_conference_network_async(df):
                # Updated plotting function
                fig = plot_author_conference_graph(G, title="Author Conference Network", 
                                                   min_shared_conferences=min_shared_conferences, 
                                                   min_citations=min_citations)

                # Ensure the application context is used when calling url_for()
                os.makedirs("static", exist_ok=True)
                graph_path = os.path.join("static", "author_network.html")

                with current_app.app_context():
                    if fig:
                        fig.write_html(graph_path, include_plotlyjs="cdn")
                        # Generate the URL for the iframe using the application context
                        html = f'<iframe src="{url_for("static", filename="author_network.html")}" width="100%" height="600" style="border:1px solid #ddd; border-radius:5px;"></iframe>'
                        result.append(html)
            return result

        # Convert async generator to a regular iterable
        async def wrapper():
            result = await generate_graph()  # Use await directly instead of asyncio.run()
            for html in result:
                yield f"data:{html}\n\n"

        return Response(wrapper(), mimetype='text/event-stream')

    except Exception as e:
        return jsonify(success=False, error=str(e))



# ------------------------------
# Run Flask app
# ------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
