from flask import Flask, render_template, request, url_for, jsonify
import os
import asyncio

from search_module import (
    find_and_extract,
    render_search_output,
    build_citation_network_async,  # async co-citation network
    plot_citation_graph
)

app = Flask(__name__)

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
    query = request.form.get("query")
    if not query:
        return "No query provided", 400

    mode = request.form.get("mode", "sections")
    try:
        n = int(request.form.get("n", 10))
        n = max(1, n)
    except ValueError:
        n = 10

    # Fetch papers
    try:
        df = find_and_extract(query, n=n, mode=mode, print_output=False)
    except Exception as e:
        return f"Error while fetching papers: {e}", 500

    html_table = render_search_output(df)

    # ------------------------------
    # Render search results page (graph off by default)
    # ------------------------------
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
        <h1>Search Results for '{query.strip()}'</h1>
        <p>Mode: <strong>{mode}</strong> | Number of results: <strong>{n}</strong></p>

        <div class="mt-4">{html_table}</div>

        <div class="form-check form-switch mt-4">
            <input class="form-check-input" type="checkbox" id="toggle_network">
            <label class="form-check-label" for="toggle_network">Show Co-Citation Network</label>
        </div>

        <div id="spinner"></div>
        <div id="graph_container"></div>

        <a href="/" class="btn btn-secondary mt-4 mb-5">← Back</a>

        <script>
            const toggle = document.getElementById('toggle_network');
            const spinner = document.getElementById('spinner');
            const container = document.getElementById('graph_container');

            toggle.addEventListener('change', async () => {{
                if (toggle.checked) {{
                    spinner.style.display = 'block';
                    container.innerHTML = '';
                    try {{
                        const formData = new FormData();
                        formData.append('query', '{query}');
                        formData.append('mode', '{mode}');
                        formData.append('n', '{n}');
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
def network():
    query = request.form.get("query")
    mode = request.form.get("mode", "sections")
    try:
        n = int(request.form.get("n", 10))
        n = max(1, n)
    except ValueError:
        n = 10

    try:
        df = find_and_extract(query, n=n, mode=mode, print_output=False)
        G = asyncio.run(build_citation_network_async(df))
        fig = plot_citation_graph(G, title="Co-Citation Network")

        os.makedirs("static", exist_ok=True)
        graph_path = os.path.join("static", "citation_graph.html")
        if fig:
            fig.write_html(graph_path, include_plotlyjs="cdn")
            html = f'<iframe src="{url_for("static", filename="citation_graph.html")}" width="100%" height="600" style="border:1px solid #ddd; border-radius:5px;"></iframe>'
            return jsonify(success=True, html=html)
        else:
            return jsonify(success=False, error="No valid co-citation graph could be generated.")
    except Exception as e:
        return jsonify(success=False, error=str(e))

# ------------------------------
# Run Flask app
# ------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
