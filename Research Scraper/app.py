import os
import asyncio
from flask import Flask, render_template, request, jsonify, Response, url_for, current_app

from search_module import (
    find_and_extract,
    render_search_output,
    build_author_conference_network_async,  # correct name
    plot_author_conference_graph              # updated plotting function
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
                        formData.append('query', '{query}');
                        formData.append('mode', '{mode}');
                        formData.append('n', '{n}');
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
# ------------------------------
# Network endpoint for toggle
# ------------------------------
@app.route("/network", methods=["POST"])
async def network():
    query = request.form.get("query")
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
        df = find_and_extract(query, n=n, mode=mode, print_output=False)
        
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
