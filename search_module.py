# search_module.py
import os
import re
import requests
import fitz  # PyMuPDF
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
from datetime import datetime
from urllib.parse import quote
import aiohttp
import asyncio

# --- Optional / Advanced (can comment out if not using async fetch) ---
# import aiohttp  # For async OpenAlex API calls (optional)

OPENALEX_URL = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/"

def search_openalex(query, n=10, extra_results=100):
    """
    Search OpenAlex using simple text search.
    """
    params = {
        "search": query,
        "per_page": extra_results
    }

    try:
        r = requests.get(OPENALEX_URL, params=params, timeout=30)
        r.raise_for_status()
        results = r.json().get("results", [])

        # Boost papers with query in title
        def relevance_score(paper):
            title = paper.get("display_name", "").lower()
            abstract = paper.get("abstract", "").lower() if paper.get("abstract") else ""
            score = 0
            if query.lower() in title:
                score += 3
            if query.lower() in abstract:
                score += 1
            return score

        # Sort by relevance, then citations as tie-breaker
        results_sorted = sorted(
            results,
            key=lambda x: (relevance_score(x), x.get("cited_by_count", 0)),
            reverse=True
        )

        return results_sorted[:n]

    except requests.RequestException as e:
        print("Error while searching OpenAlex:", e)
        return []

def get_pdf_link(paper):
    arxiv_id = paper.get("ids", {}).get("arxiv")
    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id.split('/')[-1]}.pdf"
    return paper.get("primary_location", {}).get("landing_page_url")

def download_pdf(url, filename):
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        content_type = r.headers.get('Content-Type', '')
        if 'pdf' not in content_type.lower():
            return None
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return filename
    except Exception:
        return None

def extract_full_text_from_pdf(path):
    try:
        with fitz.open(path) as doc:
            return "".join([page.get_text("text") + "\n" for page in doc])
    except Exception:
        return None

def extract_sections_from_pdf(path):
    text = extract_full_text_from_pdf(path)
    if not text:
        return {}
    sections = {}
    for sec in ["abstract", "introduction", "methods", "materials and methods"]:
        pattern = re.compile(
            rf"(?i)\b{sec}\b[\s:]*([\s\S]*?)(?=\n[A-Z][^\n]{{0,60}}\n|$)"
        )
        match = pattern.search(text)
        if match:
            sections[sec.lower()] = match.group(1).strip()
    return sections

def get_semantic_scholar_abstract(doi_or_title):
    try:
        if doi_or_title.startswith("10."):
            url = SEMANTIC_SCHOLAR_URL + f"DOI:{doi_or_title}?fields=title,abstract"
        else:
            search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            res = requests.get(search_url, params={"query": doi_or_title, "limit": 1})
            data = res.json().get("data", [])
            if not data:
                return None
            paper_id = data[0]["paperId"]
            url = SEMANTIC_SCHOLAR_URL + f"{paper_id}?fields=title,abstract"
        res = requests.get(url, timeout=15)
        if res.status_code == 200:
            return res.json().get("abstract")
    except Exception:
        pass
    return None

def find_and_extract(query, n=3, mode="sections", print_output=True):
    papers = search_openalex(query, n=n*4)
    results = []
    success_count = 0
    i = 0
    paper_index = 1

    current_year = datetime.now().year

    while success_count < n and i < len(papers):
        paper = papers[i]
        i += 1

        title = paper["display_name"]
        doi = paper.get("doi", "")
        citations = paper.get("cited_by_count", 0)
        pub_date = paper.get("publication_date", "N/A")
        publication_year = pub_date.split("-")[0] if pub_date != "N/A" else "N/A"

        authorships = paper.get("authorships", [])
        first_author_inst = authorships[0]["institutions"][0]["display_name"] if authorships and authorships[0]["institutions"] else "N/A"
        last_author_inst = authorships[-1]["institutions"][0]["display_name"] if authorships and authorships[-1]["institutions"] else "N/A"
        pdf_url = get_pdf_link(paper)

        # Extract last 4 years of citations
        counts_by_year = {int(c["year"]): c["cited_by_count"] for c in paper.get("counts_by_year", [])}
        last_4_years_citations = {str(year): counts_by_year.get(year, 0) for year in range(current_year-3, current_year+1)}

        text_data = {"abstract": None, "introduction": None, "methods": None, "full_text": None}
        extracted_something = False

        if mode != "notext" and pdf_url:
            filename = f"paper_{paper_index}.pdf"
            downloaded = download_pdf(pdf_url, filename)
            if downloaded and os.path.exists(filename):
                if mode == "full":
                    text_data["full_text"] = extract_full_text_from_pdf(filename)
                    extracted_something = True
                elif mode == "sections":
                    sections = extract_sections_from_pdf(filename)
                    if sections:
                        text_data.update({k: sections.get(k) for k in ["abstract", "introduction", "methods"]})
                        extracted_something = True
                try:
                    os.remove(filename)
                except Exception:
                    pass

        if not extracted_something and mode != "notext":
            abstract = get_semantic_scholar_abstract(doi or title)
            if abstract:
                text_data["abstract"] = abstract
                extracted_something = True

        result = {
            "title": title,
            "doi": doi,
            "citations_total": citations,
            "publication_date": pub_date,
            "publication_year": publication_year,
            "first_author_institution": first_author_inst,
            "last_author_institution": last_author_inst,
            "pdf_url": pdf_url,
            **last_4_years_citations,  # Add citations per year
            **text_data
        }

        results.append(result)
        success_count += 1
        paper_index += 1

        if print_output:
            print(f"[{paper_index-1}] {title} | Total Citations: {citations} | DOI: {doi or 'N/A'} | Year: {publication_year}")
            print("-" * 100)

    df = pd.DataFrame(results)
    return df

def normalize_doi(doi):
    """Clean and standardize DOI strings."""
    if not doi or not isinstance(doi, str):
        return None
    doi = doi.strip().lower()
    return re.sub(r'^https?://(dx\.)?doi\.org/', '', doi)

def build_citation_network(df, max_per_request=20):
    """
    Build an intra-dataset citation graph using OpenAlex 'referenced_works'.
    Efficiently fetches data in batches. Skips missing DOIs gracefully.
    """
    df = df.copy()
    df["norm_doi"] = df["doi"].apply(normalize_doi)
    valid_dois = [d for d in df["norm_doi"].dropna().unique() if d]

    doi_to_idx = {d: i for i, d in enumerate(df["norm_doi"]) if pd.notna(d)}
    G = nx.DiGraph()

    # Add all nodes first
    for i, row in df.iterrows():
        G.add_node(
            i,
            title=row["title"],
            doi=row.get("norm_doi"),
            citations=int(row.get("citations_total", 0)),
            year=row.get("publication_year", "N/A")
        )

    # Fetch works from OpenAlex in batches
    for i in range(0, len(valid_dois), max_per_request):
        batch = valid_dois[i:i + max_per_request]
        filter_str = "|".join(f"doi:{quote(d)}" for d in batch)
        url = f"https://api.openalex.org/works?filter={filter_str}&per-page={max_per_request}&select=id,doi,referenced_works"
        try:
            res = requests.get(url, timeout=20)
            if res.status_code != 200:
                continue
            works = res.json().get("results", [])

            for work in works:
                doi = normalize_doi(work.get("doi"))
                if not doi or doi not in doi_to_idx:
                    continue
                src_idx = doi_to_idx[doi]
                for ref in work.get("referenced_works", []):
                    try:
                        ref_data = requests.get(ref, timeout=5).json()
                        ref_doi = normalize_doi(ref_data.get("doi"))
                        if ref_doi and ref_doi in doi_to_idx:
                            tgt_idx = doi_to_idx[ref_doi]
                            G.add_edge(src_idx, tgt_idx)
                    except Exception:
                        continue
        except Exception as e:
            print(f"Error fetching batch {i}: {e}")
            continue

    return G


def plot_citation_graph(G):
    """Visualize the citation graph using Plotly + NetworkX."""
    pos = nx.spring_layout(G, k=0.8, iterations=100, seed=42)

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='gray'),
        hoverinfo='none',
        mode='lines'
    )

    node_x, node_y, node_text, node_size = [], [], [], []
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{data['title']}<br>DOI: {data.get('doi')}<br>Citations: {data.get('citations', 0)}")
        node_size.append(max(10, 5 + data.get("citations", 0)**0.5))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hovertext=node_text,
        marker=dict(
            size=node_size,
            color=node_size,
            colorscale='Viridis',
            showscale=True,
            line_width=1
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Intra-Dataset Citation Graph (via DOIs)",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=50)
        )
    )
    return fig

def render_md_dataframe(df):
    # Convert all values to string
    str_df = df.astype(str)
    
    # Get the max width of each column
    col_widths = [max(len(str_df[col][i]) for i in range(len(df))) for col in df.columns]
    col_widths = [max(len(col), w) for col, w in zip(df.columns, col_widths)]
    
    # Build header
    header = "| " + " | ".join(col.ljust(col_widths[i]) for i, col in enumerate(df.columns)) + " |"
    separator = "| " + " | ".join("-" * col_widths[i] for i in range(len(df.columns))) + " |"
    
    # Build rows
    rows = []
    for i in range(len(df)):
        row = "| " + " | ".join(str_df.iloc[i, j].ljust(col_widths[j]) for j in range(len(df.columns))) + " |"
        rows.append(row)
    
    # Combine everything
    md_table = "\n".join([header, separator] + rows)
    display(Markdown(md_table))

def render_html_dataframe(df):
    """
    Convert a pandas DataFrame to an HTML table suitable for embedding in a Flask web page.
    Truncates long text fields for better display.
    """
    import pandas as pd

    # Fill NaNs and convert all to strings
    str_df = df.fillna("").astype(str)

    # Optional: truncate very long text (like full_text, abstract, methods)
    max_len = 300  # characters
    for col in ["abstract", "introduction", "methods", "full_text"]:
        if col in str_df.columns:
            str_df[col] = str_df[col].apply(lambda x: x if len(x) <= max_len else x[:max_len] + "...")

    # Use pandas built-in HTML conversion with Bootstrap classes
    html_table = str_df.to_html(
        classes="table table-striped table-bordered",
        escape=True,
        index=False
    )

    return html_table

def generate_html_file(df, filename="search_results.html"):
    html = render_html_dataframe(df)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    return filename

def render_search_output(df):
    """
    Render a search results DataFrame as interactive HTML.
    Includes DataTables sorting/filtering and Plotly graphs
    that automatically update based on visible table rows.
    """

    import pandas as pd
    import json

    df = df.copy()
    df.insert(0, "#", range(1, len(df) + 1))  # Add paper numbering

    # Ensure numeric columns are numeric
    if "citations_total" in df.columns:
        df["citations_total"] = pd.to_numeric(df["citations_total"], errors="coerce").fillna(0).astype(int)

    # Identify year columns (like "2021", "2022", "2023", "2024")
    year_cols = [col for col in df.columns if col.isdigit()]
    for yc in year_cols:
        df[yc] = pd.to_numeric(df[yc], errors="coerce").fillna(0).astype(int)
    year_cols_sorted = sorted(year_cols)

    # Generate clean HTML
    table_html = df.to_html(
        classes="display nowrap",
        index=False,
        table_id="papers_table",
        escape=False,
        float_format=lambda x: f"{x:.0f}"
    )

    # JS needs access to column headers and years
    columns_json = json.dumps(list(df.columns))
    years_json = json.dumps(year_cols_sorted)

    html = f"""
<html>
<head>
    <!-- DataTables CSS & JS -->
    <link rel="stylesheet" type="text/css"
          href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css"/>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>

    <!-- Plotly -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h3>Search Results</h3>
    {table_html}

    <h4>Total Citations</h4>
    <div id="citation_dist" style="width:100%;height:400px;"></div>

    <h4>Citations Over Last 4 Years</h4>
    <div id="citation_last4" style="width:100%;height:400px;"></div>

    <script>
    $(document).ready(function() {{
        var table = $('#papers_table').DataTable({{
            paging: true,
            searching: true,
            order: [[2, "desc"]],
            scrollX: true
        }});

        var colNames = {columns_json};
        var yearCols = {years_json};

        // Find column indices dynamically
        var colIndex = {{
            num: colNames.indexOf("#"),
            title: colNames.indexOf("title"),
            citations: colNames.indexOf("citations_total")
        }};
        var yearIndices = yearCols.map(y => colNames.indexOf(y));

        function cleanNumber(val) {{
            if (!val) return 0;
            let cleaned = val.toString().replace(/[^0-9.-]/g, "");
            return parseFloat(cleaned) || 0;
        }}

        function updatePlots() {{
            var visibleRows = table.rows({{ search: 'applied', page: 'current' }}).data().toArray();

            if (visibleRows.length === 0) {{
                Plotly.purge('citation_dist');
                Plotly.purge('citation_last4');
                return;
            }}

            var paperNums = visibleRows.map(r => cleanNumber(r[colIndex.num]));
            var totalCitations = visibleRows.map(r => cleanNumber(r[colIndex.citations]));

            // --- Total Citations Plot ---
            Plotly.react('citation_dist', [{{
                x: paperNums,
                y: totalCitations,
                type: 'bar',
                name: 'Total Citations',
                marker: {{ color: 'skyblue' }}
            }}], {{
                title: 'Total Citations (Visible Papers)',
                xaxis: {{ title: 'Paper #' }},
                yaxis: {{ title: 'Citations' }}
            }});

            // --- Citations Over Last 4 Years ---
            var newTraces = [];
            visibleRows.forEach(function(r) {{
                var y_vals = yearIndices.map(i => cleanNumber(r[i]));
                newTraces.push({{
                    x: yearCols,
                    y: y_vals,
                    mode: 'lines+markers',
                    name: 'Paper ' + r[colIndex.num]
                }});
            }});

            Plotly.react('citation_last4', newTraces, {{
                title: 'Citations Over Last 4 Years (Visible Papers)',
                xaxis: {{ title: 'Year' }},
                yaxis: {{ title: 'Citations' }}
            }});
        }}

        // Initial render
        updatePlots();

        // Update on sort/search/page/length change
        table.on('order.dt search.dt page.dt length.dt', function() {{
            updatePlots();
        }});
    }});
    </script>
</body>
</html>
"""
    return html

def normalize_doi(doi):
    """Normalize DOI by removing prefix and lowercasing."""
    if not doi:
        return None
    doi = doi.strip().lower()
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]
    return doi

# -----------------------------
# Async fetch of paper metadata
# -----------------------------
async def fetch_paper_metadata(session, openalex_id):
    """
    Fetch paper metadata from OpenAlex API, especially referenced_works.
    """
    url = f"https://api.openalex.org/works/{openalex_id}"
    try:
        async with session.get(url, timeout=20) as resp:
            if resp.status != 200:
                return {}
            return await resp.json()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return {}

# -----------------------------
# Build Citation Network (bibliographic coupling)
# -----------------------------
async def build_citation_network_async(df):
    """
    Build a weighted network where papers are connected based on shared references.
    Weight = number of shared references between two papers.
    """
    df = df.copy()
    df["norm_doi"] = df["doi"].apply(normalize_doi)
    G = nx.Graph()

    # Add nodes
    for i, row in df.iterrows():
        G.add_node(
            i,
            title=row.get("title", f"Paper {i}"),
            doi=row.get("norm_doi"),
            citations=int(row.get("citations_total", 0) or 0),
            year=row.get("publication_year", "N/A")
        )

    # Fetch references for all papers asynchronously
    async with aiohttp.ClientSession() as session:
        tasks = []
        for row in df.itertuples():
            # Use OpenAlex ID if available
            openalex_id = getattr(row, "id", None) or getattr(row, "openalex_id", None)
            if openalex_id:
                tasks.append(fetch_paper_metadata(session, openalex_id))
            else:
                tasks.append(asyncio.sleep(0, result={}))

        results = await asyncio.gather(*tasks)

    # Store references per paper
    paper_refs = []
    for data in results:
        refs = set()
        for ref in data.get("referenced_works", []):
            if ref:  # ensure not None
                refs.add(ref.lower())
        paper_refs.append(refs)

    # Build edges based on shared references
    n = len(df)
    for i in range(n):
        for j in range(i + 1, n):
            shared_refs = paper_refs[i] & paper_refs[j]
            weight = len(shared_refs)
            if weight > 0:
                G.add_edge(i, j, weight=weight)

    return G

# -----------------------------
# Plot Citation Graph
# -----------------------------
def plot_citation_graph(G, title="Shared References Citation Network"):
    """
    Visualize weighted network based on shared references.
    Node size = total weight of edges (total shared references).
    Edge thickness = number of shared references.
    """
    if G.number_of_nodes() == 0:
        print("⚠️ No nodes found in citation graph.")
        return None

    # Layout
    pos = nx.spring_layout(G, k=0.8, iterations=150, seed=42)

    # Edges
    edge_x, edge_y, edge_width = [], [], []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_width.append(max(0.5, data.get("weight", 1)))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(120,120,120,0.5)'),
        hoverinfo='none',
        mode='lines'
    )

    # Nodes
    node_x, node_y, node_text, node_size = [], [], [], []
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        total_weight = sum(G[node][nbr]["weight"] for nbr in G[node])
        node_size.append(max(10, 5 + total_weight ** 0.8))

        node_text.append(
            f"<b>{data['title']}</b><br>"
            f"DOI: {data.get('doi')}<br>"
            f"Total shared refs: {total_weight}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hovertext=node_text,
        marker=dict(
            size=node_size,
            color=node_size,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Shared References"),
            line_width=1
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=title,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=50)
                    ))
    return fig

# -----------------------------
# Helper for Flask
# -----------------------------
def build_citation_network(df):
    """Sync wrapper for Flask"""
    return asyncio.run(build_citation_network_async(df))