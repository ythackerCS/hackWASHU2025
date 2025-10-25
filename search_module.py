# search_module.py

import requests, fitz, os, re
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
# from IPython.display import Markdown

OPENALEX_URL = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/"

def search_openalex(query, n=10, sort_by="cited_by_count"):
    """Search OpenAlex and return sorted results by a field (default: citations)."""
    params = {
        "search": query,
        "per_page": n,
        "sort": f"{sort_by}:desc"  # descending order
    }
    r = requests.get(OPENALEX_URL, params=params, timeout=30)
    r.raise_for_status()
    results = r.json().get("results", [])
    return results

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
    papers = search_openalex(query, n=n*4, sort_by="cited_by_count")
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

def render_search_output(df, include_graphs=True):
    """
    Render a search results DataFrame as HTML with summary stats and optional graphs.
    
    Parameters:
        df (pd.DataFrame): The results from find_and_extract.
        include_graphs (bool): Whether to include citation graphs.
    
    Returns:
        str: HTML string ready to embed in Flask.
    """
    from io import BytesIO
    import base64
    import matplotlib.pyplot as plt

    # ---------- Summary stats ----------
    total_papers = len(df)
    total_citations = df['citations_total'].sum() if 'citations_total' in df.columns else 0
    avg_citations = df['citations_total'].mean() if 'citations_total' in df.columns else 0

    summary_html = f"""
    <div class="mb-3">
        <h4>Summary Stats</h4>
        <ul>
            <li>Total papers: {total_papers}</li>
            <li>Total citations: {total_citations}</li>
            <li>Average citations per paper: {avg_citations:.2f}</li>
        </ul>
    </div>
    """

    # ---------- Table ----------
    table_html = render_html_dataframe(df)

    # ---------- Graphs ----------
    graphs_html = ""
    if include_graphs and 'citations_total' in df.columns:
        # Example: Histogram of citations
        plt.figure(figsize=(6,4))
        plt.hist(df['citations_total'], bins=10, color='skyblue', edgecolor='black')
        plt.title("Citation Distribution")
        plt.xlabel("Citations")
        plt.ylabel("Number of papers")

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        graphs_html += f'<div class="mb-3"><h4>Citation Histogram</h4><img src="data:image/png;base64,{img_base64}" class="img-fluid"></div>'

    # Example: Citations over last 4 years (if columns exist)
    year_cols = [col for col in df.columns if col.isdigit()]
    if include_graphs and year_cols:
        plt.figure(figsize=(6,4))
        for idx, row in df.iterrows():
            y = [row[year] for year in sorted(year_cols)]
            plt.plot(sorted(year_cols), y, marker='o', label=row['title'][:30] + '...')
        plt.xlabel("Year")
        plt.ylabel("Citations")
        plt.title("Citations over Last 4 Years")
        plt.legend(fontsize=8)
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        graphs_html += f'<div class="mb-3"><h4>Citations Over Years</h4><img src="data:image/png;base64,{img_base64}" class="img-fluid"></div>'

    # ---------- Combine all ----------
    final_html = f"""
    <div>
        {summary_html}
        {table_html}
        {graphs_html}
    </div>
    """
    return final_html
