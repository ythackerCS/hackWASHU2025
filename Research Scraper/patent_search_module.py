from google.cloud import bigquery
from google.oauth2 import service_account

import re
import os
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_dt
import html

def _normalize_phrase(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def create_ranked_query(search_term: str, return_count: int):
    norm_phrase = _normalize_phrase(search_term)
    norm_terms = [norm_phrase, norm_phrase.replace(' ', '')]

    SQL = r"""
    -- StandardSQL
    CREATE TEMP FUNCTION norm(s STRING) AS (
      REGEXP_REPLACE(LOWER(TRIM(s)), r'[^a-z0-9]+', ' ')
    );

    CREATE TEMP FUNCTION cos_sim(a ARRAY<FLOAT64>, b ARRAY<FLOAT64>) AS (
      SAFE_DIVIDE(
        (SELECT SUM(ax * bx)
           FROM UNNEST(a) AS ax WITH OFFSET i
           JOIN UNNEST(b) AS bx WITH OFFSET j ON i=j),
        SQRT((SELECT SUM(ax*ax) FROM UNNEST(a) AS ax)) *
        SQRT((SELECT SUM(bx*bx) FROM UNNEST(b) AS bx))
      )
    );

    WITH base AS (
      SELECT
        gpr.publication_number,
        gpr.title,
        gpr.abstract,
        gpr.url,
        gpr.embedding_v1,
        ARRAY(SELECT norm(t) FROM UNNEST(gpr.top_terms) AS t) AS tt_norm,
        SAFE.PARSE_DATE('%Y%m%d', CAST(pub.publication_date AS STRING)) AS pub_date,

        -- Use '.name' for both inventor_harmonized and assignee_harmonized
        (SELECT TRIM(inv.name)
           FROM UNNEST(pub.inventor_harmonized) AS inv WITH OFFSET i
           ORDER BY i LIMIT 1) AS lead_inventor,
        (SELECT STRING_AGG(TRIM(inv.name), '; ' ORDER BY i)
           FROM UNNEST(pub.inventor_harmonized) AS inv WITH OFFSET i) AS inventors,
        (SELECT TRIM(ass.name)
           FROM UNNEST(pub.assignee_harmonized) AS ass WITH OFFSET j
           ORDER BY j LIMIT 1) AS lead_assignee

      FROM `patents-public-data.google_patents_research.publications` AS gpr
      JOIN `patents-public-data.patents.publications` AS pub
      USING (publication_number)
      WHERE pub.country_code = 'US'
    ),

    features AS (
      SELECT
        publication_number, title, abstract, url, embedding_v1, pub_date,
        lead_inventor, inventors, lead_assignee,
        (SELECT COUNT(1) FROM UNNEST(tt_norm) AS t WHERE t IN UNNEST(@norm_terms)) AS exact_terms,
        (SELECT COUNT(1) FROM UNNEST(tt_norm) AS t WHERE t LIKE CONCAT('%', @norm_phrase, '%')) AS contains_terms,
        IF(REGEXP_CONTAINS(norm(title),    CONCAT(r'(^| )', @norm_phrase, r'( |$)')), 1, 0) AS phrase_in_title,
        IF(REGEXP_CONTAINS(norm(abstract), CONCAT(r'(^| )', @norm_phrase, r'( |$)')), 1, 0) AS phrase_in_abstract
      FROM base
    ),

    filtered AS (
      SELECT * FROM features
      WHERE exact_terms > 0 OR contains_terms > 0 OR phrase_in_title = 1 OR phrase_in_abstract = 1
    ),

    centroid AS (
      SELECT ARRAY_AGG(avg_val ORDER BY i) AS center_vec
      FROM (
        SELECT i, AVG(val) AS avg_val
        FROM filtered f, UNNEST(f.embedding_v1) AS val WITH OFFSET i
        GROUP BY i
      )
    ),

    ranked AS (
      SELECT
        f.publication_number,
        f.title,
        f.abstract,
        f.url,
        f.pub_date,
        f.lead_inventor,
        f.inventors,
        f.lead_assignee,
        cos_sim(f.embedding_v1, c.center_vec) AS cos_sim_to_centroid,
        DATE_DIFF(CURRENT_DATE(), f.pub_date, DAY) AS age_days,
        EXP(-0.15 * SAFE_DIVIDE(DATE_DIFF(CURRENT_DATE(), f.pub_date, DAY), 365.25)) AS recency_decay,
        f.exact_terms, f.contains_terms, f.phrase_in_title, f.phrase_in_abstract,
        (
          2.5 * cos_sim(f.embedding_v1, c.center_vec) +
          2.0 * f.phrase_in_title +
          1.5 * f.phrase_in_abstract +
          1.2 * f.exact_terms +
          0.6 * f.contains_terms +
          0.5 * EXP(-0.15 * SAFE_DIVIDE(DATE_DIFF(CURRENT_DATE(), f.pub_date, DAY), 365.25))
        ) AS score
      FROM filtered f
      CROSS JOIN centroid c
    )

    SELECT
      publication_number,
      url,
      title,
      abstract,
      lead_inventor,
      inventors,
      lead_assignee,
      score,
      cos_sim_to_centroid,
      exact_terms, contains_terms, phrase_in_title, phrase_in_abstract,
      age_days, recency_decay,
      pub_date AS publication_date
    FROM ranked
    ORDER BY score DESC, cos_sim_to_centroid DESC, publication_number
    LIMIT @return_count
    """

    params = [
        bigquery.ScalarQueryParameter("norm_phrase", "STRING", norm_phrase),
        bigquery.ArrayQueryParameter("norm_terms", "STRING", norm_terms),
        bigquery.ScalarQueryParameter("return_count", "INT64", int(return_count)),
    ]
    return SQL, params

# Simple normalization keys for dedupe/grouping
def _norm_name(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def dedupe_title_inventor_keep_oldest(df: pd.DataFrame, sim_eps: float = 0.005) -> pd.DataFrame:
    """
    Drops younger duplicates when:
      - same 'title' (exact string match)
      - same 'lead_inventor' (normalized)
      - abs(cos_sim_to_centroid difference) <= sim_eps
    Keeps the oldest (earliest publication_date) in each near-duplicate cluster.
    """
    d = df.copy()

    # Ensure normalized inventor key
    if "lead_inventor_norm" not in d.columns:
        d["lead_inventor_norm"] = d["lead_inventor"].map(_norm_name)

    # Robust date parsing
    if "publication_date" not in d.columns:
        raise ValueError("Expected 'publication_date' column")
    if not is_dt(d["publication_date"]):
        # handles DATE, string, or int YYYYMMDD
        d["publication_date"] = pd.to_datetime(d["publication_date"], errors="coerce", format="%Y-%m-%d")
        # fallback if ints like 20240315 slipped through
        mask_bad = d["publication_date"].isna() & d["publication_number"].notna()
        if mask_bad.any():
            try:
                as_str = df.loc[mask_bad, "publication_date"].astype(str)
                d.loc[mask_bad, "publication_date"] = pd.to_datetime(as_str, format="%Y%m%d", errors="coerce")
            except Exception:
                pass

    # Within each (title, inventor) group, sort by oldest first,
    # then drop later rows whose cos_sim_to_centroid is within sim_eps of the kept anchor.
    def _filter_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(["publication_date", "cos_sim_to_centroid"], ascending=[True, False]).reset_index(drop=True)
        kept = []
        anchor_sim = None
        anchor_date = None
        for _, row in g.iterrows():
            if anchor_sim is None:
                kept.append(True)
                anchor_sim = float(row["cos_sim_to_centroid"])
                anchor_date = row["publication_date"]
            else:
                # only drop if it's newer than anchor AND similarity is very close to anchor
                is_newer = pd.notna(anchor_date) and pd.notna(row["publication_date"]) and (row["publication_date"] >= anchor_date)
                is_similar = abs(float(row["cos_sim_to_centroid"]) - anchor_sim) <= sim_eps
                if is_newer and is_similar:
                    kept.append(False)  # drop newer near-duplicate
                else:
                    kept.append(True)
                    # reset anchor to this row (start a new cluster)
                    anchor_sim = float(row["cos_sim_to_centroid"])
                    anchor_date = row["publication_date"]
        return g[pd.Series(kept, index=g.index)]

    out = (
        d.groupby(["title", "lead_inventor_norm"], group_keys=False)
         .apply(_filter_group)
         .reset_index(drop=True)
    )
    return out

# if __name__ == "__main__":
#     os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "secrets/bq_sa.json"

#     PROJECT_ID = "brave-inn-245919"     # or cfg["gcp_project_id"]
#     LOCATION   = "US"                   # BigQuery location/region
#     SA_PATH    = "secrets/bq_sa.json"   # cfg["gcp_service_account_json"]

#     creds  = service_account.Credentials.from_service_account_file(SA_PATH)
#     client = bigquery.Client(project=PROJECT_ID, location=LOCATION, credentials=creds)

#     SQL, params = create_ranked_query("brain-computer interface", 20)
#     df_ranked = client.query(SQL, job_config=bigquery.QueryJobConfig(query_parameters=params)).to_dataframe()
#     df_ranked = df_ranked.drop(columns=["exact_terms","contains_terms","phrase_in_title","phrase_in_abstract"], errors="ignore")
#     df_dedup = dedupe_title_inventor_keep_oldest(df_ranked, sim_eps=0.005)
#     df_dedup = df_dedup.drop(['inventors'], axis=1)

def search_patents_df(
    search_term: str,
    n_results: int,
    *,
    project_id: str = "brave-inn-245919",
    sa_path: str = "secrets/bq_sa.json",
    location: str = "US",
    sim_eps: float = 0.005
) -> str:
    """
    Run the ranked patent query and return a HTML table (string).
    Assumes `create_ranked_query` and `dedupe_title_inventor_keep_oldest` are available.
    """
    creds = service_account.Credentials.from_service_account_file(sa_path)
    client = bigquery.Client(project=project_id, location=location, credentials=creds)

    SQL, params = create_ranked_query(search_term, n_results)
    df = client.query(SQL, job_config=bigquery.QueryJobConfig(query_parameters=params)).to_dataframe()

    # Drop intermediate flags (if present)
    df = df.drop(
        columns=["exact_terms", "contains_terms", "phrase_in_title", "phrase_in_abstract"],
        errors="ignore"
    )

    # Deduplicate and tidy
    df = dedupe_title_inventor_keep_oldest(df, sim_eps=sim_eps)
    df = df.drop(columns=["inventors"], errors="ignore")

    # Make URL column clickable if it exists
    if "url" in df.columns:
        df["url"] = df["url"].apply(
            lambda u: f'<a href="{html.escape(u, quote=True)}" target="_blank">link</a>'
                      if isinstance(u, str) and u else ""
        )

    # Optional: move URL next to title if present
    col_order = [c for c in ["title", "url"] if c in df.columns] + [c for c in df.columns if c not in {"title","url"}]
    df = df[col_order]

    return df
    
def render_patent_search_output(df):
    """
    Render a patent search DataFrame as interactive HTML.
    - Only Title is clickable.
    - Robust to url values that are already <a ...>link</a> (extracts the real href).
    Expected columns:
      title, url, publication_number, abstract, lead_inventor, lead_assignee,
      score, cos_sim_to_centroid, age_days, recency_decay, publication_date
    """
    import pandas as pd
    import numpy as np
    import html, json, re

    df = df.copy()

    # Ensure columns exist
    expected = [
        "title","url","publication_number","abstract","lead_inventor","lead_assignee",
        "score","cos_sim_to_centroid","age_days","recency_decay","publication_date"
    ]
    for c in expected:
        if c not in df.columns:
            df[c] = np.nan

    # Normalize types
    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce").dt.date.astype(str)
    for c in ["score", "cos_sim_to_centroid", "recency_decay"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["age_days"] = pd.to_numeric(df["age_days"], errors="coerce").astype("Int64")

    # --- URL hygiene ---
    href_re = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)

    def extract_href(u: str) -> str:
        """Return the URL string; if u is an <a …>…</a>, pull its href."""
        if not isinstance(u, str):
            return ""
        u = u.strip()
        if not u:
            return ""
        m = href_re.search(u)
        return (m.group(1).strip() if m else u)

    def title_as_link(title, url):
        t = html.escape(str(title) if title is not None else "")
        raw = extract_href(url)
        if raw:
            safe = html.escape(raw, quote=True)
            return f'<a href="{safe}" target="_blank" rel="noopener noreferrer">{t}</a>'
        return t

    def abstract_cell(a):
        if not isinstance(a, str) or not a.strip():
            return ""
        short = html.escape(a[:180]) + ("..." if len(a) > 180 else "")
        full  = html.escape(a)
        return (f'<details><summary>{short}</summary>'
                f'<div style="white-space:pre-wrap;margin-top:.5rem;">{full}</div></details>')

    # Build output table (Pub # stays plain text)
    out = pd.DataFrame({
        "#": range(1, len(df) + 1),
        "Title": [title_as_link(t, u) for t, u in zip(df["title"], df["url"])],
        "Pub #": df["publication_number"].fillna("").astype(str),
        "Assignee": df["lead_assignee"].fillna(""),
        "Lead Inventor": df["lead_inventor"].fillna(""),
        "Pub Date": df["publication_date"].fillna(""),
        "Age (days)": df["age_days"].astype("string").fillna(""),
        "Score": df["score"].map(lambda x: f"{x:.3f}" if pd.notna(x) else ""),
        "Cosine to Centroid": df["cos_sim_to_centroid"].map(lambda x: f"{x:.3f}" if pd.notna(x) else ""),
        "Recency Decay": df["recency_decay"].map(lambda x: f"{x:.3f}" if pd.notna(x) else ""),
        "Abstract": df["abstract"].map(abstract_cell),
    })

    table_html = out.to_html(
        classes="display nowrap compact",
        index=False,
        table_id="patents_table",
        escape=False,   # keep <a> and <details>
        border=0
    )

    columns_json = json.dumps(list(out.columns))

    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Patent Results</title>
  <link rel="stylesheet" type="text/css"
        href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css"/>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 1.5rem; }}
    table.dataTable thead th {{ white-space: nowrap; }}
    details > summary {{ cursor: pointer; color: #0d6efd; }}
    details[open] > summary {{ color: #0a58ca; }}
  </style>
  <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
  <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
</head>
<body>
  <h3>Patent Search Results</h3>
  {table_html}

  <script>
  $(function() {{
      const colNames = {columns_json};
      const scoreIdx = colNames.indexOf("Score");
      const order = (scoreIdx >= 0) ? [[scoreIdx, "desc"]] : [[0, "asc"]];
      $('#patents_table').DataTable({{
          paging: true,
          searching: true,
          order: order,
          scrollX: true,
          responsive: true,
          pageLength: 10,
          lengthMenu: [10, 25, 50, 100]
      }});
  }});
  </script>
</body>
</html>
"""
