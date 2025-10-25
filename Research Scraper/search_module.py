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
from urllib.parse import quote  # for URL encoding
import urllib.request  # if you need request functionality
import aiohttp
import asyncio
import itertools  # for combinations of authors
from collections import defaultdict  # for author -> conferences mapping
import json  # if you need JSON serialization for Flask responses
import time 
import random


# --- Optional / Advanced (can comment out if not using async fetch) ---
# import aiohttp  # For async OpenAlex API calls (optional)

OPENALEX_URL = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/"

def search_openalex(query, n=10, extra_results=100):
    """
    Search OpenAlex using simple text search.
    Returns list of dicts with first/last author names + schools.
    """
    import requests

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

        # Extract first/last author names and schools
        for paper in results:
            authorships = paper.get("authorships", [])
            if authorships:
                # First author
                first = authorships[0]
                first_name = first.get("author", {}).get("display_name", "")
                first_aff = first.get("institutions", [])
                first_school = first_aff[0]["display_name"] if first_aff else ""
                paper["first_author"] = f"{first_name} | {first_school}"

                # Last author
                last = authorships[-1]
                last_name = last.get("author", {}).get("display_name", "")
                last_aff = last.get("institutions", [])
                last_school = last_aff[0]["display_name"] if last_aff else ""
                paper["last_author"] = f"{last_name} | {last_school}"
            else:
                paper["first_author"] = ""
                paper["last_author"] = ""

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

        # --- First & Last Author with School ---
        authorships = paper.get("authorships", [])
        if authorships:
            # First author
            first_author_name = authorships[0].get("author", {}).get("display_name", "N/A")
            first_aff = authorships[0].get("institutions", [])
            first_author_school = first_aff[0]["display_name"] if first_aff else "N/A"
            first_author_full = f"{first_author_name} | {first_author_school}"

            # Last author
            last_author_name = authorships[-1].get("author", {}).get("display_name", "N/A")
            last_aff = authorships[-1].get("institutions", [])
            last_author_school = last_aff[0]["display_name"] if last_aff else "N/A"
            last_author_full = f"{last_author_name} | {last_author_school}"
        else:
            first_author_full = "N/A"
            last_author_full = "N/A"

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
            "first_author": first_author_full,
            "last_author": last_author_full,
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


def plot_citation_graph(G, title="Co-Citation Network"):
    """Visualize the citation graph using Plotly + NetworkX."""
    import networkx as nx
    import plotly.graph_objects as go

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
        # Wrap long titles for display
        title_text = data.get('title', '')
        if len(title_text) > 60:
            title_text = title_text[:57] + "..."
        node_text.append(f"{title_text}<br>DOI: {data.get('doi')}<br>Citations: {data.get('citations_total', 0)}")
        node_size.append(max(10, 5 + (data.get("citations_total", 0) ** 0.5)))

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
                        title=title,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=50, l=50, r=50, t=100)
                    ))
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
    Combines first/last author name + institution for display.
    Includes DataTables sorting/filtering and Plotly graphs.
    """

    import pandas as pd
    import json

    df = df.copy()

    # Combine author name + institution
    if all(col in df.columns for col in ["first_author_name", "first_author_institution"]):
        df['first_author'] = df.apply(
            lambda r: f"{r['first_author_name']} | {r['first_author_institution']}", axis=1
        )
    if all(col in df.columns for col in ["last_author_name", "last_author_institution"]):
        df['last_author'] = df.apply(
            lambda r: f"{r['last_author_name']} | {r['last_author_institution']}", axis=1
        )

    # Drop the separate columns to avoid clutter
    for col in ["first_author_name", "first_author_institution", "last_author_name", "last_author_institution"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Add paper numbering
    df.insert(0, "#", range(1, len(df) + 1))

    # Ensure numeric columns are numeric
    if "citations_total" in df.columns:
        df["citations_total"] = pd.to_numeric(df["citations_total"], errors="coerce").fillna(0).astype(int)

    # Identify year columns (like "2021", "2022", "2023", "2024")
    year_cols = [col for col in df.columns if col.isdigit()]
    for yc in year_cols:
        df[yc] = pd.to_numeric(df[yc], errors="coerce").fillna(0).astype(int)
    year_cols_sorted = sorted(year_cols)

    # Generate clean HTML table
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
    <link rel="stylesheet" type="text/css"
          href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css"/>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
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
            scrollX: true,
            responsive: true
        }});

        var colNames = {columns_json};
        var yearCols = {years_json};

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

            // Total Citations Bar
            Plotly.react('citation_dist', [{{
                x: paperNums,
                y: totalCitations,
                type: 'bar',
                name: 'Total Citations',
                marker: {{ color: 'skyblue' }},
                text: visibleRows.map(r => r[colNames.indexOf("first_author")] + '<br>' + r[colNames.indexOf("last_author")]),
                hoverinfo: 'x+y+text'
            }}], {{
                title: 'Total Citations (Visible Papers)',
                xaxis: {{ title: 'Paper #' }},
                yaxis: {{ title: 'Citations' }}
            }});

            // Last 4 Years Line Plot
            var newTraces = [];
            visibleRows.forEach(function(r) {{
                var y_vals = yearIndices.map(i => cleanNumber(r[i]));
                newTraces.push({{
                    x: yearCols,
                    y: y_vals,
                    mode: 'lines+markers',
                    name: 'Paper ' + r[colIndex.num] + ': ' + r[colNames.indexOf("title")],
                    text: r[colNames.indexOf("first_author")] + '<br>' + r[colNames.indexOf("last_author")],
                    hoverinfo: 'name+x+y+text'
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

        // Update on table change
        table.on('order.dt search.dt page.dt length.dt', function() {{
            updatePlots();
        }});
    }});
    </script>
</body>
</html>
"""
    return html

prestige_rank = {
    "NeurIPS": 10,
    "CVPR": 9,
    "ICML": 9,
    "ECCV": 8,
    "ICLR": 8,
    "AAAI": 7,
    "IJCAI": 7,
    "SIGGRAPH": 10,  # High prestige for computer graphics
    "MICCAI": 8,
    # Add more conferences and their rankings...
}

author_cache = {}

async def fetch_author_metadata(session, author_name, institution_name=None, attempt=1):
    OPENALEX_URL = "https://api.openalex.org"
    """
    Fetch additional metadata (e.g., citation count) for an author using OpenAlex.
    This function applies exponential backoff when rate limited.
    """
    if author_name in author_cache:
        print(f"Using cached data for {author_name}")
        return author_cache[author_name]  # Return cached data
    
    # Extract only the author's name (remove affiliation part)
    author_name_only = author_name.split(' | ')[0]  # Everything before the first " | "
    
    try:
        # Base query URL for searching authors
        query_url = f"{OPENALEX_URL}/authors?search={urllib.parse.quote(author_name_only)}"
        
        # If institution_name is provided, refine the query
        if institution_name:
            query_url += f"&filter=affiliations.institution.display_name:{urllib.parse.quote(institution_name)}"
        
        # Making the request with SSL verification disabled
        async with session.get(query_url, ssl=False) as response:  # ssl=False to disable certificate verification
            # Handle Rate Limiting (HTTP 429)
            if response.status == 429:
                reset_time = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))  # Default to 60 seconds
                sleep_time = reset_time - time.time()
                if sleep_time > 0:
                    # Exponential backoff with random jitter
                    wait_time = min(2 ** attempt + random.uniform(0, 1), 180)  # Increased max wait time to 180 seconds
                    print(f"Rate limited. Sleeping for {wait_time:.2f} seconds before retrying for {author_name_only}")
                    await asyncio.sleep(wait_time)
                    return await fetch_author_metadata(session, author_name, institution_name, attempt + 1)  # Retry with increased attempt

            # Handle Access Denied (HTTP 403)
            elif response.status == 403:
                print(f"Access forbidden for {author_name_only}. Skipping.")
                return {'author': author_name_only, 'citations': 0}

            # Handle Not Found (HTTP 404) and other errors
            if response.status != 200:
                print(f"Error fetching data for {author_name_only}: {response.status}")
                return {'author': author_name_only, 'citations': 0}

            # Parse JSON response
            data = await response.json()

            if not data.get("results"):
                print(f"No results found for {author_name_only}.")
                return {'author': author_name_only, 'citations': 0}

            # Extract citation data
            total_citations = data['results'][0].get('cited_by_count', 0)
            
            # Cache the result
            result = {'author': author_name_only, 'citations': total_citations}
            author_cache[author_name_only] = result
            return result

    except Exception as e:
        print(f"Error while fetching metadata for {author_name_only}: {e}")
        return {'author': author_name_only, 'citations': 0}


# ------------------------------
# Build Author Conference Network and update incrementally
async def build_author_conference_network_async(df):
    """
    Build an async author-level network based on conference participation.
    Node size = # conferences
    Edge weight = # shared prestigious conferences (by year)
    """
    df = df.copy()
    G = nx.Graph()
    author_confs = defaultdict(lambda: defaultdict(set))  # author -> {year -> set(conferences)}

    # Collect authors -> conferences mapping
    for _, row in df.iterrows():
        authors = []
        if row.get("first_author") and row["first_author"] != "N/A":
            authors.append(row["first_author"])
        if row.get("last_author") and row["last_author"] != "N/A":
            authors.append(row["last_author"])

        conf_name = row.get("venue") or row.get("conference") or "Unknown Venue"
        year = str(row.get("publication_year", "N/A"))

        for author in authors:
            author_confs[author][year].add(conf_name)

    # Exit early if no authors found
    if len(author_confs) == 0:
        print("⚠️ No authors found in DataFrame.")
        yield G
        return

    print(f"DEBUG: Total authors collected: {len(author_confs)}")

    # Optionally fetch additional metadata asynchronously (e.g., citations) for the top authors
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_author_metadata(session, author) for author in author_confs.keys()]
        results = await asyncio.gather(*tasks)

    # Add nodes for authors with citation and shared conference filters
    author_metadata = {result['author']: result for result in results}

    skipped_authors = []

    # Process authors in batches and update the graph incrementally
    batch_size = 10  # Update the graph after processing every 10 authors
    all_authors = list(author_confs.items())
    for i in range(0, len(all_authors), batch_size):
        batch = all_authors[i:i + batch_size]

        # Add nodes for the current batch of authors
        for author, confs_by_year in batch:
            total_confs = sum(len(confs) for confs in confs_by_year.values())
            citations = author_metadata.get(author, {}).get('citations', 0)
            
            # Add the author to the graph
            G.add_node(
                author,
                conferences=total_confs,
                years=list(confs_by_year.keys()),
                citations=citations,
                metadata=author_metadata.get(author, {})
            )

        # Return the updated graph after every batch
        print(f"DEBUG: Added {len(batch)} authors to the graph.")
        
        # Yield the graph incrementally so the frontend can update
        yield G  # This is where we yield the updated graph to show progress

    # Once all authors are processed, yield the final graph
    print("All authors processed.")
    yield G  # Yield the final graph


# ------------------------------
# Plotting the Author Conference Network without filtering
def plot_author_conference_graph(
    G, 
    title="Author Conference Network", 
    min_shared_conferences=0, 
    min_citations=0
):
    """
    Plot the author-conference network with optional filters:
      - min_shared_conferences: only show edges with at least this many shared conferences
      - min_citations: only show authors with at least this many total conferences/citations
    """
    if G.number_of_nodes() == 0:
        print("⚠️ No authors found.")
        return None

    # Step 1: Remove the filtering by min_shared_conferences (no threshold)
    H = nx.Graph()
    for u, v, data in G.edges(data=True):
        H.add_edge(u, v, **data)

    # Step 2: Remove the filtering by min_citations (no threshold)
    for node, data in G.nodes(data=True):
        H.add_node(node, **data)

    if H.number_of_nodes() == 0:
        print(f"⚠️ No authors found after processing the graph.")
        return None

    pos = nx.spring_layout(H, k=0.8, iterations=150, seed=42)

    # Edges
    edge_x, edge_y = [], []
    for u, v, data in H.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(120,120,120,0.5)'),
        hoverinfo='none',
        mode='lines'
    )

    # Nodes
    node_x, node_y, node_text, node_size = [], [], [], []
    for node, data in H.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_size.append(max(10, data.get("conferences", 1) * 5))
        node_text.append(
            f"<b>{node}</b><br>"
            f"Conferences: {data.get('conferences', 0)}<br>"
            f"Years active: {', '.join(data.get('years', []))}"
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
            colorbar=dict(title="Conference Count"),
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


def build_author_conference_network(df):
    """Sync wrapper for Flask"""
    return asyncio.run(build_author_conference_network_async(df))