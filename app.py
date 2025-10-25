from flask import Flask, render_template, request
from search_module import find_and_extract, render_html_dataframe, render_search_output


app = Flask(__name__)

# Home page showing the search form
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")  # Form points to /search

# Search endpoint handling form submission
@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query")
    if not query:
        return "No query provided", 400

    # Get mode and n from form
    mode = request.form.get("mode", "sections")
    try:
        n = int(request.form.get("n", 10))
        if n < 1:
            n = 10
    except ValueError:
        n = 10

    try:
        # Run your search with user-selected mode and n
        df = find_and_extract(query, n=n, mode=mode, print_output=False)
    except Exception as e:
        return f"Error while searching: {e}", 500

    # Convert DataFrame to HTML
    html_table = render_search_output(df, include_graphs=True)

    # Render results in a simple HTML page
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Search Results</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
    </head>
    <body class="container mt-5">
        <h1>Search Results for '{query}'</h1>
        <p>Mode: {mode} | Number of results: {n}</p>
        <div class="mt-4">{html_table}</div>
        <a href="/" class="btn btn-secondary mt-3">Back</a>
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
