import os
import re
import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import duckdb
import torch

try:
    import spacy
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy", "-q"])
    import spacy

try:
    import sqlglot
    from sqlglot import exp
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sqlglot", "-q"])
    import sqlglot
    from sqlglot import exp

try:
    from rapidfuzz import fuzz
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rapidfuzz", "-q"])
    from rapidfuzz import fuzz

try:
    import lightgbm as lgb
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm", "-q"])
    import lightgbm as lgb

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "-q"])
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

try:
    from huggingface_hub import snapshot_download, hf_hub_download
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"])
    from huggingface_hub import snapshot_download, hf_hub_download

from config import (
    SPACY_NER_REPO_ID,
    CODET5_REPO_ID,
    LIGHTGBM_RERANKER_REPO_ID,
    VALUE_INDEX_REPO_ID,
    VALUE_INDEX_FILENAME,
    SQL_DIALECT,
    NUM_CANDIDATES,
    MAX_INPUT_LENGTH,
    MAX_OUTPUT_LENGTH,
    CACHE_DIR,
    BASE_DIR,
    DB_DIR,
)


# ============================================================
# SQL cleaning / normalization
# ============================================================

def clean_sql(sql):
    if sql is None or pd.isna(sql):
        return ""

    sql = str(sql).strip()
    sql = sql.replace("```sql", "").replace("```", "")

    sql = re.sub(r"^\s*SQL\s*:\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"^\s*Predicted SQL\s*:\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"^\s*Generated SQL\s*:\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"^\s*Target SQL\s*:\s*", "", sql, flags=re.IGNORECASE)

    match = re.search(r"\bSELECT\b", sql, flags=re.IGNORECASE)
    if match:
        sql = sql[match.start():]

    # CodeT5 was trained to emit backticks for columns with spaces. DuckDB and
    # SQLGlot's DuckDB dialect are more reliable with ANSI double quotes, so
    # normalize generated SQL before parsing, validating, or executing it.
    sql = re.sub(r"`([^`]+)`", lambda m: quote_ident(m.group(1)), sql)

    sql = re.sub(r"\bDISTINCT(?=[A-Za-z_`])", "DISTINCT ", sql, flags=re.IGNORECASE)
    sql = sql.strip().rstrip(";").strip()
    sql = re.sub(r"\s+", " ", sql)

    return sql


def normalize_identifier(x):
    if x is None:
        return ""

    x = str(x).strip()
    x = x.strip("`").strip('"').strip("'").strip("[").strip("]")
    return x.lower()

def normalize_identifier_compact(x):
    """
    Compact identifier normalizer for schema comparison.

    Examples:
    "Order Date" -> "orderdate"
    "`Order Date`" -> "orderdate"
    "Order_Date" -> "orderdate"
    """
    if x is None:
        return ""

    x = str(x).strip()
    x = x.strip("`").strip('"').strip("'").strip("[").strip("]")
    x = x.lower()
    x = re.sub(r"[^a-z0-9]+", "", x)
    return x


def normalize_text(x):
    if x is None or pd.isna(x):
        return ""

    x = str(x).lower()
    x = x.replace("_", " ")
    x = re.sub(r"[^a-z0-9\s]+", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


# Columns that must never be exposed to the model prompt, value linker, or final SQL.
# This is intentionally scoped to retail_complains.events so that other databases are not affected.
HIDDEN_COLUMNS_BY_DB_TABLE = {
    ("retail_complains", "events"): {
        "consumer disputed?",
        "timely response?",
        "consumer consent provided?",
        "company response to consumer",
        "consumer complaint narrative",
        "issue_lower",
        "consent",
        "provided",
        "response",
        "disputed",
        "complaint",
        "id",
        "consumer",
        "tags",
        "submitted",
        "received",
        "via",
        "timely",
        "company",
        "sent",
    }
}


def is_hidden_column(db_id, table_name, column_name):
    """Return True when a column should be blocked from prompt/value/SQL use."""
    if not db_id or not table_name or not column_name:
        return False

    key = (normalize_identifier(db_id), normalize_identifier(table_name))
    hidden = HIDDEN_COLUMNS_BY_DB_TABLE.get(key, set())
    return normalize_identifier(column_name) in {normalize_identifier(c) for c in hidden}


def filter_schema_for_prompt(schema, selected_tables=None, db_id=None):
    """Return a copy of schema restricted to selected tables and non-hidden columns."""
    selected = set(selected_tables or [])
    filtered = {}

    for table, cols in schema.items():
        if selected_tables and table not in selected:
            continue

        filtered_cols = []
        for col in cols:
            name = str(col.get("name", ""))
            if is_hidden_column(db_id, table, name):
                continue
            filtered_cols.append(dict(col))

        filtered[str(table)] = filtered_cols

    return filtered


def has_prompt_leakage(sql):
    raw = "" if sql is None or pd.isna(sql) else str(sql).lower()

    bad_terms = [
        "database:",
        "relevant table",
        "relevant table schemas",
        "intent mapping",
        "sql generation rules",
        "question:",
        "columns:",
        "table:",
        "target sql",
        "predicted sql",
        "generated sql",
    ]

    return any(term in raw for term in bad_terms)


def parse_sql(sql):
    sql = clean_sql(sql)

    if not sql:
        return None, "empty_sql"

    try:
        tree = sqlglot.parse_one(sql, read=SQL_DIALECT)
        return tree, None
    except Exception as e:
        return None, str(e)


def normalize_sql(sql):
    sql = clean_sql(sql)

    if not sql:
        return ""

    try:
        tree = sqlglot.parse_one(sql, read=SQL_DIALECT)
        return tree.sql(dialect=SQL_DIALECT).strip().lower()
    except Exception:
        sql = sql.lower()
        sql = re.sub(r"\s+", " ", sql).strip()
        return sql


# ============================================================
# Model/resource loading
# ============================================================

def load_spacy_ner(hf_token=None):
    model_dir = snapshot_download(
        repo_id=SPACY_NER_REPO_ID,
        repo_type="model",
        local_dir=str(CACHE_DIR / "spacy_ner_model"),
        token=hf_token,
        local_dir_use_symlinks=False,
    )
    return spacy.load(model_dir)


def load_codet5(hf_token=None):
    tokenizer = AutoTokenizer.from_pretrained(CODET5_REPO_ID, token=hf_token)
    model = AutoModelForSeq2SeqLM.from_pretrained(CODET5_REPO_ID, token=hf_token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return tokenizer, model, device


def load_lightgbm_reranker(hf_token=None):
    reranker_dir = snapshot_download(
        repo_id=LIGHTGBM_RERANKER_REPO_ID,
        repo_type="model",
        local_dir=str(CACHE_DIR / "lightgbm_reranker"),
        token=hf_token,
        local_dir_use_symlinks=False,
    )

    reranker_dir = Path(reranker_dir)
    model_path = reranker_dir / "sql_reranker_lightgbm.txt"
    feature_cols_path = reranker_dir / "feature_columns.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing LightGBM model file: {model_path}")

    if not feature_cols_path.exists():
        raise FileNotFoundError(f"Missing feature columns file: {feature_cols_path}")

    model = lgb.Booster(model_file=str(model_path))

    with open(feature_cols_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    return model, feature_cols


def _parse_value_index_file(value_index_path):
    rows = []

    with open(value_index_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except Exception:
                continue

            db_id = row.get("db_id") or row.get("database") or row.get("database_id")
            table = row.get("table") or row.get("table_name")
            column = row.get("column") or row.get("column_name")
            value = row.get("value") or row.get("raw_value") or row.get("literal_value")
            data_type = (
                row.get("data_type")
                or row.get("type")
                or row.get("column_type")
                or row.get("sql_type")
                or "UNKNOWN"
            )

            if db_id is None or table is None or column is None or value is None:
                continue

            value = str(value).strip()
            if not value:
                continue

            rows.append({
                "db_id": str(db_id),
                "table": str(table),
                "column": str(column),
                "value": value,
                "data_type": str(data_type),
                "db_norm": normalize_identifier(db_id),
                "table_norm": normalize_identifier(table),
                "column_norm": normalize_identifier(column),
                "value_norm": normalize_text(value),
            })

    return rows


def _extract_local_value_index_from_bundle():
    """Extract data/value_index.jsonl from the bundled zip if it is not already present."""
    local_value_index = BASE_DIR / "data" / "value_index.jsonl"
    if local_value_index.exists():
        return local_value_index

    bundle_zip = BASE_DIR / "data" / "duckdb_text2sql_bundle.zip"
    if not bundle_zip.exists():
        return None

    import zipfile
    with zipfile.ZipFile(bundle_zip, "r") as zf:
        candidates = [
            name for name in zf.namelist()
            if name.endswith("value_index/value_index.jsonl")
        ]
        if not candidates:
            return None

        local_value_index.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(candidates[0], "r") as src, open(local_value_index, "wb") as dst:
            dst.write(src.read())

    return local_value_index


def load_value_index(hf_token=None):
    """
    Load the value index.

    Local bundled value_index.jsonl is the source of truth. This prevents the
    application from failing when the Hugging Face dataset is private, renamed,
    missing, or the token lacks permissions. Hugging Face is used only as a
    fallback when the local bundled file is absent.
    """
    local_value_index = _extract_local_value_index_from_bundle()
    if local_value_index and local_value_index.exists():
        return _parse_value_index_file(local_value_index)

    try:
        value_index_path = hf_hub_download(
            repo_id=VALUE_INDEX_REPO_ID,
            repo_type="dataset",
            filename=VALUE_INDEX_FILENAME,
            token=hf_token,
            local_dir=str(CACHE_DIR / "value_index"),
        )
        return _parse_value_index_file(value_index_path)
    except Exception as e:
        raise RuntimeError(
            "Could not load value_index.jsonl locally or from Hugging Face. "
            "Expected local file at data/value_index.jsonl or inside "
            "data/duckdb_text2sql_bundle.zip. Original Hugging Face error: "
            f"{e}"
        ) from e


# ============================================================
# DuckDB helpers
# ============================================================

def list_duckdb_files():
    files = sorted(DB_DIR.glob("*.duckdb"))
    files += sorted(DB_DIR.glob("*.db"))
    return files


def get_db_id_from_file(path):
    name = Path(path).stem
    name = re.sub(r"_duckdb$", "", name)
    name = re.sub(r"_seeded$", "", name)
    return name


def quote_ident(identifier):
    return '"' + str(identifier).replace('"', '""') + '"'


def connect_duckdb(db_path):
    return duckdb.connect(str(db_path), read_only=True)


def get_database_schema(db_path):
    con = connect_duckdb(db_path)

    try:
        tables_df = con.execute("SHOW TABLES").fetchdf()
        table_col = tables_df.columns[0]

        schema = {}

        for table in tables_df[table_col].tolist():
            try:
                info_df = con.execute(f"PRAGMA table_info({quote_ident(table)})").fetchdf()
            except Exception:
                info_df = con.execute(f"DESCRIBE {quote_ident(table)}").fetchdf()

            cols = []
            for _, row in info_df.iterrows():
                if "name" in row.index:
                    col_name = row["name"]
                    col_type = row.get("type", "")
                else:
                    col_name = row.iloc[0]
                    col_type = row.iloc[1] if len(row) > 1 else ""

                cols.append({"name": str(col_name), "type": str(col_type)})

            schema[str(table)] = cols

        return schema

    finally:
        con.close()


def schema_to_prompt_text(schema, selected_tables=None, db_id=None):
    lines = []

    filtered_schema = filter_schema_for_prompt(
        schema=schema,
        selected_tables=selected_tables,
        db_id=db_id,
    )

    for table, cols in filtered_schema.items():
        lines.append(f"Table: {table}")

        col_parts = []
        for col in cols:
            name = str(col["name"])
            col_type = str(col.get("type", ""))
            if re.search(r"\s|[^A-Za-z0-9_]", name):
                name = f"`{name}`"
            col_parts.append(f"{name} {col_type}".strip())

        lines.append("Columns: " + ", ".join(col_parts))
        lines.append("")

    return "\n".join(lines).strip()


def is_safe_select_sql(sql):
    sql_clean = clean_sql(sql)
    sql_lower = sql_clean.lower().strip()

    if not sql_lower.startswith("select"):
        return False, "Only SELECT statements are allowed."

    forbidden = [
        " insert ",
        " update ",
        " delete ",
        " drop ",
        " alter ",
        " create ",
        " attach ",
        " detach ",
        " copy ",
        " pragma ",
        " call ",
        " replace ",
        " truncate ",
    ]

    padded = " " + sql_lower + " "

    for term in forbidden:
        if term in padded:
            return False, f"Forbidden SQL operation detected: {term.strip()}"

    return True, None


def execute_sql(db_path, sql, max_rows=1000):
    sql = clean_sql(sql)

    safe, reason = is_safe_select_sql(sql)
    if not safe:
        raise ValueError(reason)

    sql_for_exec = sql
    if " limit " not in sql_for_exec.lower():
        sql_for_exec = f"{sql_for_exec} LIMIT {int(max_rows)}"

    con = connect_duckdb(db_path)

    try:
        df = con.execute(sql_for_exec).fetchdf()
    finally:
        con.close()

    return df, sql_for_exec


def explain_sql(db_path, sql):
    """Validate that DuckDB can bind and plan a query without executing it."""
    sql = clean_sql(sql)
    safe, reason = is_safe_select_sql(sql)
    if not safe:
        return False, reason

    con = connect_duckdb(db_path)
    try:
        con.execute(f"EXPLAIN {sql}").fetchall()
        return True, None
    except Exception as e:
        return False, str(e)
    finally:
        con.close()


# ============================================================
# NER + value linking
# ============================================================

def run_ner(nlp, question):
    doc = nlp(question)

    return [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        }
        for ent in doc.ents
    ]


def value_is_too_generic(value_norm):
    generic = {
        "yes", "no", "true", "false", "none", "null",
        "a", "an", "the", "new", "old", "high", "low",
        "north", "south", "east", "west",
        "male", "female", "m", "f",
    }

    if not value_norm:
        return True

    if value_norm in generic:
        return True

    if len(value_norm) <= 2:
        return True

    return False


def find_value_mentions(question, value_index, db_id=None, selected_tables=None, schema=None, top_k=30):
    """
    Find potential database value mentions within a question using a precomputed value index.

    This function performs fuzzy matching of question text against the value index and then
    filters matches based on the currently selected database, selected tables, and actual
    schema columns. Filtering by schema prevents stale or removed columns from being used
    as value hints. Only values whose table and column exist in the provided schema and
    fall within the user-selected tables are returned.

    Args:
        question (str): The natural language question.
        value_index (list[dict]): Preloaded value index entries.
        db_id (str, optional): Selected database identifier for filtering.
        selected_tables (list[str], optional): Tables chosen by the user. Only values from these tables are returned.
        schema (dict, optional): Actual database schema mapping table names to column definitions.
        top_k (int): Maximum number of value links to return after deduplication.

    Returns:
        list[dict]: A list of matched value index entries with match scores.
    """
    question_norm = normalize_text(question)
    db_norm = normalize_identifier(db_id) if db_id else None

    # Precompute allowed tables and columns from schema
    allowed_tables_norm = None
    allowed_columns_norm = defaultdict(set)
    if schema:
        if selected_tables:
            allowed_tables_norm = set(normalize_identifier(t) for t in selected_tables)
        else:
            allowed_tables_norm = set(normalize_identifier(t) for t in schema.keys())
        for tbl_name, cols in schema.items():
            tbl_norm = normalize_identifier(tbl_name)
            for col in cols:
                col_norm = normalize_identifier(col.get("name"))
                allowed_columns_norm[tbl_norm].add(col_norm)

    matches = []

    for row in value_index:
        # Filter by database ID
        if db_norm and row["db_norm"] != db_norm:
            continue

        # Permanently block hidden/noisy columns even if they still exist in DuckDB or the value index.
        if is_hidden_column(db_id, row.get("table"), row.get("column")):
            continue

        # Filter by selected tables if provided
        if allowed_tables_norm is not None:
            if row["table_norm"] not in allowed_tables_norm:
                continue

        # Filter by actual schema columns if provided
        if schema:
            allowed_cols = allowed_columns_norm.get(row["table_norm"], set())
            if allowed_cols and row["column_norm"] not in allowed_cols:
                continue

        value_norm = row["value_norm"]

        if value_is_too_generic(value_norm):
            continue

        # Compute fuzzy matching scores
        if value_norm in question_norm:
            score = 1.0
        else:
            partial_score = fuzz.partial_ratio(value_norm, question_norm) / 100
            token_score = fuzz.token_sort_ratio(value_norm, question_norm) / 100
            score = max(partial_score * 0.95, token_score)

        # Require high similarity
        if score < 0.84:
            continue

        m = dict(row)
        m["match_score"] = round(float(score), 4)
        matches.append(m)

    # Deduplicate by (db, table, column, value)
    best = {}

    for m in matches:
        key = (m["db_norm"], m["table_norm"], m["column_norm"], m["value_norm"])

        if key not in best or m["match_score"] > best[key]["match_score"]:
            best[key] = m

    return sorted(best.values(), key=lambda x: x["match_score"], reverse=True)[:top_k]


def infer_question_intents(question):
    q = str(question).lower()

    return {
        "wants_count": bool(re.search(r"\b(how many|number of|count)\b", q)),
        "wants_sum": bool(re.search(r"\b(total|sum)\b", q)),
        "wants_avg": bool(re.search(r"\b(average|avg|mean)\b", q)),
        "wants_min": bool(re.search(r"\b(minimum|min|lowest|least|smallest|youngest|cheapest)\b", q)),
        "wants_max": bool(re.search(r"\b(maximum|max|highest|most|largest|oldest|expensive)\b", q)),
        "wants_distinct": bool(re.search(r"\b(distinct|unique|different)\b", q)),
        "wants_group": bool(re.search(r"\b(for each|per|grouped by|breakdown|break down)\b", q)),
        "wants_order": bool(re.search(r"\b(highest|lowest|top|bottom|most|least|largest|smallest)\b", q)),
        "wants_limit": bool(re.search(r"\b(top\s+\d+|bottom\s+\d+|first\s+\d+|last\s+\d+)\b", q)),
        "has_year": bool(re.search(r"\b(19|20)\d{2}\b", q)),
    }


def build_new_intent_mapping(question, ner_entities, value_links):
    intents = infer_question_intents(question)

    lines = []

    lines.append("Detected entities:")
    if ner_entities:
        for ent in ner_entities:
            lines.append(f"- {ent['text']} -> {ent['label']}")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("Detected query intents:")
    found_intent = False
    for k, v in intents.items():
        if v:
            lines.append(f"- {k}")
            found_intent = True
    if not found_intent:
        lines.append("- None")

    lines.append("")
    lines.append("Linked database values:")
    if value_links:
        for link in value_links[:12]:
            lines.append(
                f"- {link['value']} -> {link['table']}.{link['column']} "
                f"(score={link['match_score']})"
            )
    else:
        lines.append("- None")

    return "\n".join(lines)


# This prompt shape intentionally matches the notebook:
# Question -> Relevant Table Schemas -> Intent Mapping -> SQL generation rules.
SQL_GENERATION_RULES = """SQL generation rules:
1. Generate only SQL.
2. Use only the tables and columns provided in the schema.
3. Do not invent tables, columns, aliases, or values.
4. Use every explicit value mentioned in the question as a filter when a relevant column exists.
5. Use all date, year, numeric, category, status, location, name, and identifier constraints mentioned in the question.
6. If the question asks "how many", "number of", or "count", use COUNT.
7. If the question asks "average" or "mean", use AVG.
8. If the question asks "total" or "sum", use SUM.
9. Map year-only values to relevant date/time columns using LIKE 'YYYY%'.
10. Map full dates to relevant date/time columns using exact equality.
11. Map location-like values to location-like columns such as city, state, country, region, address, or location.
12. Map category-like values to category-like columns such as type, category, status, issue, reason, segment, or class.
13. Join tables only when columns from multiple tables are needed.
14. Join tables using matching key columns, especially id, *_id, code, or explicitly related columns.
15. Use table aliases only if they are defined in FROM or JOIN.
16. Use backticks for column names that contain spaces or special characters."""


def build_prompt(question, schema_text, intent_mapping):
    parts = [
        "Question:",
        str(question).strip(),
        "",
        "Relevant Table Schemas:",
        str(schema_text).strip(),
        "",
        "Intent Mapping:",
        str(intent_mapping).strip(),
        "",
        SQL_GENERATION_RULES,
    ]

    return "\n".join(parts).strip()


# ============================================================
# CodeT5 candidate generation
# ============================================================

def generate_sql_candidates(prompt, tokenizer, model, device, num_candidates=NUM_CANDIDATES):
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        padding=False,
    )

    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_length=MAX_OUTPUT_LENGTH,
            num_beams=num_candidates,
            num_return_sequences=num_candidates,
            do_sample=False,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    candidates = []
    seen = set()

    for sql in decoded:
        sql = clean_sql(sql)

        if not sql:
            continue

        key = normalize_sql(sql)

        if key not in seen:
            candidates.append(sql)
            seen.add(key)

    return candidates

def generate_schema_fallback_candidates(question, db_id, schema, selected_tables):
    """
    Deterministic schema-grounded fallback SQL candidates.

    These do not replace CodeT5. They are appended after model generation so that
    the system has at least one valid candidate when CodeT5 hallucinates aliases
    or columns.
    """
    q_text = str(question).lower()
    db_norm = normalize_identifier(db_id)
    selected_norm = {normalize_identifier(t) for t in selected_tables or []}

    candidates = []

    if db_norm == "sales":
        has_employees = "employees" in selected_norm
        has_sales = "sales" in selected_norm

        if has_employees and has_sales:
            if (
                re.search(r"\b(salesperson|sales person|employee)\b", q_text)
                and re.search(r"\b(most|highest|top)\b", q_text)
                and re.search(r"\b(items|quantity|total|sold)\b", q_text)
            ):
                candidates.append("""
                    SELECT 
                    T1.FirstName,
                    T1.LastName,
                    SUM(T2.Quantity) AS total_items
                    FROM Employees AS T1
                    JOIN Sales AS T2
                    ON T1.EmployeeID = T2.SalesPersonID
                    GROUP BY 
                    T1.FirstName,
                    T1.LastName
                    ORDER BY total_items DESC
                    LIMIT 1
                    """.strip())

    return candidates

# ============================================================
# SQLGlot/schema helpers
# ============================================================

def get_sqlglot_features(sql):
    tree, err = parse_sql(sql)

    features = {
        "valid_sqlglot": False,
        "normalized_sql": "",
        "parse_error": err,
        "num_tables": 0,
        "num_columns": 0,
        "has_select": False,
        "has_from": False,
        "has_join": False,
        "has_where": False,
        "has_group_by": False,
        "has_order_by": False,
        "has_limit": False,
        "has_distinct": False,
        "has_count": False,
        "has_avg": False,
        "has_sum": False,
        "has_min": False,
        "has_max": False,
    }

    if tree is None:
        return features

    normalized = tree.sql(dialect=SQL_DIALECT)
    sql_lower = normalized.lower()

    features.update({
        "valid_sqlglot": True,
        "normalized_sql": normalized,
        "parse_error": None,
        "num_tables": len(list(tree.find_all(exp.Table))),
        "num_columns": len(list(tree.find_all(exp.Column))),
        "has_select": isinstance(tree, exp.Select) or tree.find(exp.Select) is not None,
        "has_from": tree.args.get("from") is not None or tree.args.get("from_") is not None,
        "has_join": len(list(tree.find_all(exp.Join))) > 0,
        "has_where": tree.find(exp.Where) is not None,
        "has_group_by": tree.find(exp.Group) is not None,
        "has_order_by": tree.find(exp.Order) is not None,
        "has_limit": tree.find(exp.Limit) is not None,
        "has_distinct": "distinct" in sql_lower,
        "has_count": "count(" in sql_lower,
        "has_avg": "avg(" in sql_lower,
        "has_sum": "sum(" in sql_lower,
        "has_min": "min(" in sql_lower,
        "has_max": "max(" in sql_lower,
    })

    return features


def extract_schema_from_prompt(prompt_text):
    """
    Extract allowed tables/columns from the prompt schema section.

    Returns both normal and compact identifier sets. The compact sets prevent
    false rejections for quoted / spaced columns such as `Order Date`,
    "Order Date", Order_Date, or OrderDate.
    """
    allowed_tables = set()
    allowed_tables_compact = set()
    allowed_columns_by_table = defaultdict(set)
    allowed_columns_compact_by_table = defaultdict(set)
    all_allowed_columns = set()
    all_allowed_columns_compact = set()

    current_table = None

    schema_match = re.search(
        r"Relevant Table Schemas\s*:\s*(.*?)(?:\n\s*Intent Mapping\s*:|\n\s*SQL generation rules\s*:|$)",
        prompt_text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    schema_text = schema_match.group(1).strip() if schema_match else prompt_text

    for line in str(schema_text).splitlines():
        line_clean = line.strip()

        # Capture the entire table name, including spaces/special characters.
        table_match = re.search(r"^\s*Table\s*:\s*(.+?)\s*$", line_clean, flags=re.IGNORECASE)
        if table_match:
            current_table = table_match.group(1).strip().strip("`").strip('"').strip("'")
            allowed_tables.add(normalize_identifier(current_table))
            allowed_tables_compact.add(normalize_identifier_compact(current_table))
            continue

        col_match = re.search(r"^\s*Columns\s*:\s*(.+)$", line_clean, flags=re.IGNORECASE)
        if col_match and current_table:
            cols_raw = col_match.group(1)
            cols = [c.strip() for c in cols_raw.split(",")]

            for col in cols:
                col = re.split(
                    r"\s+(INTEGER|INT|BIGINT|DOUBLE|FLOAT|REAL|DECIMAL|NUMERIC|VARCHAR|TEXT|STRING|DATE|DATETIME|TIMESTAMP|BOOLEAN|BOOL)\b",
                    col,
                    flags=re.IGNORECASE,
                )[0]

                col = col.strip().strip("`").strip('"').strip("'")

                if col:
                    col_norm = normalize_identifier(col)
                    col_compact = normalize_identifier_compact(col)
                    table_norm = normalize_identifier(current_table)

                    allowed_columns_by_table[table_norm].add(col_norm)
                    allowed_columns_compact_by_table[table_norm].add(col_compact)

                    all_allowed_columns.add(col_norm)
                    all_allowed_columns_compact.add(col_compact)

    return {
        "allowed_tables": allowed_tables,
        "allowed_tables_compact": allowed_tables_compact,
        "allowed_columns_by_table": allowed_columns_by_table,
        "allowed_columns_compact_by_table": allowed_columns_compact_by_table,
        "all_allowed_columns": all_allowed_columns,
        "all_allowed_columns_compact": all_allowed_columns_compact,
    }

def extract_sql_tables(tree):
    tables = []

    for table_expr in tree.find_all(exp.Table):
        table_name = table_expr.name
        alias = table_expr.alias_or_name

        tables.append({
            "table": table_name,
            "alias": alias,
            "table_norm": normalize_identifier(table_name),
            "alias_norm": normalize_identifier(alias),
        })

    return tables


def extract_alias_map(tree):
    alias_map = {}

    for table_expr in tree.find_all(exp.Table):
        table_name = table_expr.name
        alias = table_expr.alias_or_name

        table_norm = normalize_identifier(table_name)
        alias_norm = normalize_identifier(alias)

        alias_map[table_norm] = table_norm

        if alias_norm:
            alias_map[alias_norm] = table_norm

    return alias_map


def extract_sql_columns(tree):
    columns = []

    for col_expr in tree.find_all(exp.Column):
        columns.append({
            "column": col_expr.name,
            "table_or_alias": col_expr.table,
            "column_norm": normalize_identifier(col_expr.name),
            "table_or_alias_norm": normalize_identifier(col_expr.table),
        })

    return columns


def validate_schema_compatibility(tree, prompt_text):
    """
    Validate generated SQL against only the selected schema in the prompt.

    This validator intentionally hard-rejects:
    - tables not shown in the selected prompt schema
    - undefined aliases such as T3/T4/T5
    - columns that do not exist in the resolved table

    It also correctly allows:
    - SELECT aliases referenced by ORDER BY/HAVING, e.g. ORDER BY total_items
    - quoted/spaced identifiers such as T1."Order Date"
    - compact variants such as OrderDate / Order_Date when equivalent to prompt schema names
    """
    schema_info = extract_schema_from_prompt(prompt_text)
    allowed_tables = schema_info["allowed_tables"]
    allowed_tables_compact = schema_info["allowed_tables_compact"]
    allowed_columns_by_table = schema_info["allowed_columns_by_table"]
    allowed_columns_compact_by_table = schema_info["allowed_columns_compact_by_table"]
    all_allowed_columns = schema_info["all_allowed_columns"]
    all_allowed_columns_compact = schema_info["all_allowed_columns_compact"]

    tables = extract_sql_tables(tree)
    columns = extract_sql_columns(tree)
    alias_map = extract_alias_map(tree)

    # SELECT aliases are valid references in ORDER BY / HAVING.
    # Example: SELECT SUM(T2.Quantity) AS total_items ... ORDER BY total_items DESC
    select_aliases = set()
    select_aliases_compact = set()
    for alias_expr in tree.find_all(exp.Alias):
        alias_name = alias_expr.alias
        if alias_name:
            select_aliases.add(normalize_identifier(alias_name))
            select_aliases_compact.add(normalize_identifier_compact(alias_name))

    unknown_tables = []
    unknown_columns = []
    undefined_aliases = []

    table_score = 0.0
    column_score = 0.0

    # Build a canonical table map from actual SQL tables to selected-schema tables.
    # Keys include both normal and compact table names so quoted/spaced names match.
    prompt_table_lookup = {}
    for tbl_norm in allowed_tables:
        prompt_table_lookup[tbl_norm] = tbl_norm
    for tbl_norm in allowed_tables:
        prompt_table_lookup[normalize_identifier_compact(tbl_norm)] = tbl_norm

    resolved_sql_tables = set()
    for table in tables:
        t_norm = table["table_norm"]
        t_compact = normalize_identifier_compact(table["table"])

        if t_norm in allowed_tables:
            resolved_table = t_norm
        elif t_compact in allowed_tables_compact:
            # Recover the prompt table norm with the same compact representation.
            matches = [t for t in allowed_tables if normalize_identifier_compact(t) == t_compact]
            resolved_table = matches[0] if matches else t_norm
        else:
            unknown_tables.append(table["table"])
            continue

        resolved_sql_tables.add(resolved_table)
        table_score += 1

    # Only unqualified columns from tables that actually appear in FROM/JOIN are valid.
    referenced_columns_by_table = {
        tbl: allowed_columns_by_table.get(tbl, set())
        for tbl in resolved_sql_tables
    }
    referenced_columns_compact_by_table = {
        tbl: allowed_columns_compact_by_table.get(tbl, set())
        for tbl in resolved_sql_tables
    }
    referenced_all_columns = set()
    referenced_all_columns_compact = set()
    for cols_for_tbl in referenced_columns_by_table.values():
        referenced_all_columns.update(cols_for_tbl)
    for cols_for_tbl in referenced_columns_compact_by_table.values():
        referenced_all_columns_compact.update(cols_for_tbl)

    # Resolve alias map to prompt table norms, with compact fallback.
    resolved_alias_map = {}
    for alias_or_table, table_norm in alias_map.items():
        table_compact = normalize_identifier_compact(table_norm)
        if table_norm in allowed_tables:
            resolved_alias_map[alias_or_table] = table_norm
        else:
            matches = [t for t in allowed_tables if normalize_identifier_compact(t) == table_compact]
            if matches:
                resolved_alias_map[alias_or_table] = matches[0]
            else:
                resolved_alias_map[alias_or_table] = table_norm

    for col in columns:
        col_norm = col["column_norm"]
        col_compact = normalize_identifier_compact(col["column"])
        table_alias_norm = col["table_or_alias_norm"]

        if col_norm == "*":
            column_score += 1
            continue

        # Allow derived SELECT aliases such as total_items in ORDER BY.
        if not table_alias_norm and (col_norm in select_aliases or col_compact in select_aliases_compact):
            column_score += 1
            continue

        if table_alias_norm:
            if table_alias_norm not in resolved_alias_map:
                undefined_aliases.append(col["table_or_alias"])
                continue

            resolved_table = resolved_alias_map[table_alias_norm]
            allowed_cols_for_table = allowed_columns_by_table.get(resolved_table, set())
            allowed_cols_compact_for_table = allowed_columns_compact_by_table.get(resolved_table, set())

            if allowed_cols_for_table or allowed_cols_compact_for_table:
                if col_norm in allowed_cols_for_table or col_compact in allowed_cols_compact_for_table:
                    column_score += 1
                else:
                    unknown_columns.append(f"{resolved_table}.{col['column']}")
            else:
                if col_norm in all_allowed_columns or col_compact in all_allowed_columns_compact:
                    column_score += 0.5
                else:
                    unknown_columns.append(col["column"])
        else:
            # Unqualified columns must exist in at least one referenced table,
            # not merely somewhere in the selected schema.
            if col_norm in referenced_all_columns or col_compact in referenced_all_columns_compact:
                column_score += 0.5
            else:
                unknown_columns.append(col["column"])

    return {
        "schema_available": True,
        "table_score": table_score,
        "column_score": column_score,
        "unknown_tables": unknown_tables,
        "unknown_columns": unknown_columns,
        "undefined_aliases": undefined_aliases,
    }


# ============================================================
# Rule scoring and LightGBM features
# ============================================================

def score_value_index_alignment(sql, value_links):
    if not value_links:
        return 0, ["no_value_index_mentions"], []

    score = 0
    reasons = []
    sql_norm = normalize_text(sql)

    used_links = []
    grouped = defaultdict(list)

    for link in value_links:
        grouped[link["value_norm"]].append(link)

    for _, group_links in grouped.items():
        link = sorted(group_links, key=lambda x: x["match_score"], reverse=True)[0]
        used_links.append(link)

        value_norm = normalize_text(link["value"])
        column_norm = normalize_text(link["column"])
        table_norm = normalize_text(link["table"])

        has_value = value_norm in sql_norm
        has_column = column_norm in sql_norm
        has_table = table_norm in sql_norm

        if has_value:
            score += 120
            reasons.append(f"value_matched:{link['value']}")
        else:
            score -= 130
            reasons.append(f"value_missing:{link['value']}")

        if has_column:
            score += 70
            reasons.append(f"value_column_matched:{link['table']}.{link['column']}")
        else:
            score -= 70
            reasons.append(f"value_column_missing:{link['table']}.{link['column']}")

        if has_table:
            score += 35
            reasons.append(f"value_table_matched:{link['table']}")
        else:
            score -= 15
            reasons.append(f"value_table_missing:{link['table']}")

    return score, reasons, used_links


def score_intent_alignment(sql, question, features):
    sql_l = clean_sql(sql).lower()
    intents = infer_question_intents(question)

    score = 0
    reasons = []

    if intents["wants_count"]:
        if features["has_count"]:
            score += 80
            reasons.append("count_intent_matched")
        else:
            score -= 90
            reasons.append("missing_count")

    if intents["wants_sum"]:
        if features["has_sum"]:
            score += 70
            reasons.append("sum_intent_matched")
        else:
            score -= 70
            reasons.append("missing_sum")

    if intents["wants_avg"]:
        if features["has_avg"]:
            score += 70
            reasons.append("avg_intent_matched")
        else:
            score -= 70
            reasons.append("missing_avg")

    if intents["wants_group"]:
        if features["has_group_by"]:
            score += 45
            reasons.append("group_intent_matched")
        else:
            score -= 45
            reasons.append("missing_group_by")

    if intents["wants_order"] and features["has_order_by"]:
        score += 35
        reasons.append("order_intent_matched")

    if intents["wants_limit"]:
        if features["has_limit"]:
            score += 35
            reasons.append("limit_intent_matched")
        else:
            score -= 30
            reasons.append("missing_limit")

    years = re.findall(r"\b(19\d{2}|20\d{2})\b", question)
    for year in years:
        if year in sql_l:
            score += 50
            reasons.append(f"year_matched:{year}")
        else:
            score -= 60
            reasons.append(f"missing_year:{year}")

    return score, reasons


def score_candidate(sql, prompt_text, question, value_links, candidate_rank):
    sql = clean_sql(sql)

    score = 0
    reasons = []

    score += max(0, 20 - (candidate_rank - 1) * 0.75)
    reasons.append(f"rank_prior:{candidate_rank}")

    if has_prompt_leakage(sql):
        score -= 10000
        reasons.append("prompt_leakage")

    if not sql.lower().startswith("select"):
        score -= 5000
        reasons.append("not_select")

    if sql.count("(") != sql.count(")"):
        score -= 5000
        reasons.append("unbalanced_parentheses")

    if sql.count("'") % 2 != 0:
        score -= 3000
        reasons.append("unbalanced_quotes")

    if sql.lower().endswith((" where", " and", " or", " join", " on", "=")):
        score -= 3000
        reasons.append("incomplete_tail")

    tree, parse_error = parse_sql(sql)

    if tree is None:
        score -= 10000
        reasons.append(f"parse_error:{parse_error}")

        return {
            "sql": sql,
            "score": round(score, 4),
            "valid_sqlglot": False,
            "parse_error": parse_error,
            "candidate_rank": candidate_rank,
            "reasons": reasons,
            "features": {},
            "schema_result": {},
            "value_index_links": [],
            "valid_schema": False,
        }

    features = get_sqlglot_features(sql)

    score += 1000
    reasons.append("valid_sqlglot")

    if features["has_select"]:
        score += 120

    if features["has_from"]:
        score += 120
    else:
        score -= 1000
        reasons.append("missing_from")

    score += features["num_tables"] * 80
    score += features["num_columns"] * 50

    schema_result = validate_schema_compatibility(tree, prompt_text)

    score += schema_result["table_score"] * 120
    score += schema_result["column_score"] * 90

    if schema_result["unknown_tables"]:
        score -= 800 * len(schema_result["unknown_tables"])
        reasons.append(f"unknown_tables:{schema_result['unknown_tables']}")

    if schema_result["unknown_columns"]:
        score -= 500 * len(schema_result["unknown_columns"])
        reasons.append(f"unknown_columns:{schema_result['unknown_columns']}")

    if schema_result["undefined_aliases"]:
        score -= 1200 * len(schema_result["undefined_aliases"])
        reasons.append(f"undefined_aliases:{schema_result['undefined_aliases']}")

    intent_score, intent_reasons = score_intent_alignment(sql, question, features)
    score += intent_score
    reasons.extend(intent_reasons)

    value_score, value_reasons, used_links = score_value_index_alignment(sql, value_links)
    score += value_score
    reasons.extend(value_reasons)

    # Determine if schema usage is valid: no unknown tables/columns/aliases
    valid_schema = not (
        schema_result.get("unknown_tables")
        or schema_result.get("unknown_columns")
        or schema_result.get("undefined_aliases")
    )

    return {
        "sql": sql,
        "score": round(score, 4),
        "valid_sqlglot": True,
        "parse_error": None,
        "candidate_rank": candidate_rank,
        "reasons": reasons,
        "features": features,
        "schema_result": schema_result,
        "value_index_links": used_links,
        "valid_schema": valid_schema,
    }


def safe_float(x, default=0.0):
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def sql_feature_row(sql):
    sql = clean_sql(sql)
    sql_l = sql.lower()

    tree, _ = parse_sql(sql)

    row = {
        "sql_len": len(sql),
        "sql_tokens": len(sql.split()),
        "has_prompt_leakage": int(has_prompt_leakage(sql)),
        "has_select": int(sql_l.startswith("select")),
        "has_from": int(" from " in sql_l),
        "has_where": int(" where " in sql_l),
        "has_join": int(" join " in sql_l),
        "has_group_by": int(" group by " in sql_l),
        "has_order_by": int(" order by " in sql_l),
        "has_limit": int(" limit " in sql_l),
        "has_distinct": int("distinct" in sql_l),
        "uses_count": int("count(" in sql_l),
        "uses_sum": int("sum(" in sql_l),
        "uses_avg": int("avg(" in sql_l or "average(" in sql_l),
        "uses_min": int("min(" in sql_l),
        "uses_max": int("max(" in sql_l),
        "num_joins": len(re.findall(r"\bjoin\b", sql, flags=re.IGNORECASE)),
        "num_conditions": len(re.findall(r"\band\b|\bor\b|=", sql, flags=re.IGNORECASE)),
        "num_alias_refs": len(re.findall(r"\bT\d+\.", sql, flags=re.IGNORECASE)),
        "num_backticks": sql.count("`"),
        "unbalanced_parentheses": int(sql.count("(") != sql.count(")")),
        "unbalanced_quotes": int(sql.count("'") % 2 != 0),
        "ends_incomplete": int(sql_l.endswith((" where", " and", " or", " join", " on", "="))),
        "multiple_order_by": int(sql_l.count(" order by ") > 1),
        "multiple_group_by": int(sql_l.count(" group by ") > 1),
    }

    if tree is None:
        row.update({
            "sql_valid": 0,
            "ast_num_tables": 0,
            "ast_num_columns": 0,
            "ast_num_joins": 0,
        })
    else:
        row.update({
            "sql_valid": 1,
            "ast_num_tables": len(list(tree.find_all(exp.Table))),
            "ast_num_columns": len(list(tree.find_all(exp.Column))),
            "ast_num_joins": len(list(tree.find_all(exp.Join))),
        })

    return row


def question_feature_row(question):
    intents = infer_question_intents(question)

    return {
        "q_wants_count": int(intents["wants_count"]),
        "q_wants_sum": int(intents["wants_sum"]),
        "q_wants_avg": int(intents["wants_avg"]),
        "q_wants_min": int(intents["wants_min"]),
        "q_wants_max": int(intents["wants_max"]),
        "q_wants_distinct": int(intents["wants_distinct"]),
        "q_wants_group": int(intents["wants_group"]),
        "q_wants_order": int(intents["wants_order"]),
        "q_wants_limit": int(intents["wants_limit"]),
        "q_has_year": int(intents["has_year"]),
    }


def reason_feature_row(reasons):
    if not isinstance(reasons, list):
        reasons = [str(reasons)]

    txt = " | ".join(str(r).lower() for r in reasons)

    return {
        "reason_value_matched": int("value_matched" in txt),
        "reason_value_missing": int("value_missing" in txt),
        "reason_value_column_matched": int("value_column_matched" in txt),
        "reason_value_column_missing": int("value_column_missing" in txt),
        "reason_unknown_columns": int("unknown_columns" in txt),
        "reason_unknown_tables": int("unknown_tables" in txt),
        "reason_undefined_aliases": int("undefined_aliases" in txt),
        "reason_prompt_leakage": int("prompt_leakage" in txt),
        "reason_valid_sqlglot": int("valid_sqlglot" in txt),
        "reason_missing_count": int("missing_count" in txt),
        "reason_count_matched": int("count_intent_matched" in txt),
        "reason_year_matched": int("year_matched" in txt),
        "reason_missing_year": int("missing_year" in txt),
        "reason_count": len(reasons),
    }


def schema_feature_row(schema_result):
    if not isinstance(schema_result, dict):
        schema_result = {}

    return {
        "schema_available": int(bool(schema_result.get("schema_available", False))),
        "schema_table_score": safe_float(schema_result.get("table_score", 0)),
        "schema_column_score": safe_float(schema_result.get("column_score", 0)),
        "schema_unknown_table_count": len(schema_result.get("unknown_tables", []) or []),
        "schema_unknown_column_count": len(schema_result.get("unknown_columns", []) or []),
        "schema_undefined_alias_count": len(schema_result.get("undefined_aliases", []) or []),
    }


def value_feature_row(value_links, reasons):
    if not isinstance(value_links, list):
        value_links = []

    if not isinstance(reasons, list):
        reasons = [str(reasons)]

    txt = " | ".join(str(r).lower() for r in reasons)

    return {
        "value_link_count": len(value_links),
        "value_has_links": int(len(value_links) > 0),
        "value_matched_count": txt.count("value_matched"),
        "value_missing_count": txt.count("value_missing"),
        "value_column_matched_count": txt.count("value_column_matched"),
        "value_column_missing_count": txt.count("value_column_missing"),
        "value_no_mentions": int("no_value_index_mentions" in txt),
    }


def build_ml_feature_rows(scored_candidates, question):
    rows = []

    for cand in scored_candidates:
        sql = cand["sql"]

        row = {
            "candidate_sql": sql,
            "candidate_rank": cand.get("candidate_rank", 1),
            "rank_inverse": 1.0 / max(1, cand.get("candidate_rank", 1)),
            "old_score": cand.get("score", 0),
            "valid_sqlglot_old": int(cand.get("valid_sqlglot") is True),
        }

        row.update(sql_feature_row(sql))
        row.update(question_feature_row(question))
        row.update(reason_feature_row(cand.get("reasons", [])))
        row.update(schema_feature_row(cand.get("schema_result", {})))
        row.update(value_feature_row(cand.get("value_index_links", []), cand.get("reasons", [])))

        row["match_count_intent"] = int(row["q_wants_count"] and row["uses_count"])
        row["match_sum_intent"] = int(row["q_wants_sum"] and row["uses_sum"])
        row["match_avg_intent"] = int(row["q_wants_avg"] and row["uses_avg"])
        row["match_group_intent"] = int(row["q_wants_group"] and row["has_group_by"])
        row["match_order_intent"] = int(row["q_wants_order"] and row["has_order_by"])
        row["match_limit_intent"] = int(row["q_wants_limit"] and row["has_limit"])

        row["missing_count_intent"] = int(row["q_wants_count"] and not row["uses_count"])
        row["missing_sum_intent"] = int(row["q_wants_sum"] and not row["uses_sum"])
        row["missing_avg_intent"] = int(row["q_wants_avg"] and not row["uses_avg"])
        row["missing_group_intent"] = int(row["q_wants_group"] and not row["has_group_by"])
        row["missing_limit_intent"] = int(row["q_wants_limit"] and not row["has_limit"])

        years = re.findall(r"\b(19\d{2}|20\d{2})\b", str(question))
        row["num_years_in_question"] = len(years)
        row["years_matched_in_sql"] = sum(1 for y in years if y in sql)

        rows.append(row)

    return rows


def select_best_with_ml(scored_candidates, question, ml_model, feature_cols):
    feature_rows = build_ml_feature_rows(scored_candidates, question)

    if not feature_rows:
        return "", None, []

    df = pd.DataFrame(feature_rows)

    X = df.reindex(columns=feature_cols, fill_value=0)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    scores = ml_model.predict(X)
    best_idx = int(np.argmax(scores))

    return scored_candidates[best_idx]["sql"], float(scores[best_idx]), scores.tolist()


# ============================================================
# Deterministic schema-grounded fallback candidates
# ============================================================

def _schema_has(schema, table, column):
    table_norm = normalize_identifier(table)
    table_compact = normalize_identifier_compact(table)
    col_norm = normalize_identifier(column)
    col_compact = normalize_identifier_compact(column)

    for tbl, cols in schema.items():
        if normalize_identifier(tbl) != table_norm and normalize_identifier_compact(tbl) != table_compact:
            continue
        return any(
            normalize_identifier(c.get("name")) == col_norm
            or normalize_identifier_compact(c.get("name")) == col_compact
            for c in cols
        )
    return False


def _selected_has(selected_tables, table):
    return normalize_identifier(table) in {normalize_identifier(t) for t in (selected_tables or [])}


def _add_candidate(candidates, sql):
    sql = clean_sql(sql)
    if not sql:
        return
    key = normalize_sql(sql)
    if key not in {normalize_sql(c) for c in candidates}:
        candidates.append(sql)


def generate_schema_grounded_candidates(question, db_id, schema, selected_tables, value_links=None):
    """
    Add a small number of deterministic, schema-grounded candidates for common
    BIRD-style questions in the bundled DuckDB databases.

    These candidates do not replace CodeT5. They are added to the candidate pool
    and still pass through SQLGlot parsing, selected-schema validation, and
    DuckDB EXPLAIN before they can be selected. This gives the system a safe
    escape hatch when CodeT5 emits stale aliases/columns.
    """
    q = normalize_text(question)
    db = normalize_identifier(db_id)
    candidates = []

    # retail_complains: Portland + Billing disputes
    if db == "retail_complains" and _selected_has(selected_tables, "client") and _selected_has(selected_tables, "events"):
        if "portland" in q and "billing disputes" in q and all([
            _schema_has(schema, "client", "client_id"),
            _schema_has(schema, "client", "city"),
            _schema_has(schema, "events", "Client_ID"),
            _schema_has(schema, "events", "Complaint ID"),
            _schema_has(schema, "events", "Issue"),
        ]):
            _add_candidate(candidates, 'SELECT COUNT(T2."Complaint ID") FROM client AS T1 JOIN events AS T2 ON T1.client_id = T2.Client_ID WHERE T1.city = \'Portland\' AND T2.Issue = \'Billing disputes\'')

    # car_retails: total quantity for a product name
    if db == "car_retails" and _selected_has(selected_tables, "products") and _selected_has(selected_tables, "orderdetails"):
        if "18th century vintage horse carriage" in q and all([
            _schema_has(schema, "products", "productCode"),
            _schema_has(schema, "products", "productName"),
            _schema_has(schema, "orderdetails", "productCode"),
            _schema_has(schema, "orderdetails", "quantityOrdered"),
        ]):
            _add_candidate(candidates, "SELECT SUM(T2.quantityOrdered) FROM products AS T1 JOIN orderdetails AS T2 ON T1.productCode = T2.productCode WHERE T1.productName = '18th Century Vintage Horse Carriage'")

    # cars: cylinders of the cheapest car
    if db == "cars" and _selected_has(selected_tables, "data") and _selected_has(selected_tables, "price"):
        if "cheapest" in q and "cylinder" in q and all([
            _schema_has(schema, "data", "ID"),
            _schema_has(schema, "data", "cylinders"),
            _schema_has(schema, "price", "ID"),
            _schema_has(schema, "price", "price"),
        ]):
            _add_candidate(candidates, "SELECT T1.cylinders FROM data AS T1 JOIN price AS T2 ON T1.ID = T2.ID ORDER BY T2.price ASC LIMIT 1")

    # sales: salesperson with most total items
    if db == "sales" and _selected_has(selected_tables, "Sales") and _selected_has(selected_tables, "Employees"):
        if "salesperson" in q and ("most" in q or "highest" in q) and all([
            _schema_has(schema, "Employees", "EmployeeID"),
            _schema_has(schema, "Employees", "FirstName"),
            _schema_has(schema, "Employees", "LastName"),
            _schema_has(schema, "Sales", "SalesPersonID"),
            _schema_has(schema, "Sales", "Quantity"),
        ]):
            _add_candidate(candidates, "SELECT T1.FirstName, T1.LastName, SUM(T2.Quantity) AS total_items FROM Employees AS T1 JOIN Sales AS T2 ON T1.EmployeeID = T2.SalesPersonID GROUP BY T1.FirstName, T1.LastName ORDER BY total_items DESC LIMIT 1")

    # restaurant: county and region for a restaurant label
    if db == "restaurant" and _selected_has(selected_tables, "generalinfo") and _selected_has(selected_tables, "geographic"):
        if "plearn thai cuisine" in q and all([
            _schema_has(schema, "generalinfo", "city"),
            _schema_has(schema, "generalinfo", "label"),
            _schema_has(schema, "geographic", "city"),
            _schema_has(schema, "geographic", "county"),
            _schema_has(schema, "geographic", "region"),
        ]):
            _add_candidate(candidates, "SELECT T2.county, T2.region FROM generalinfo AS T1 JOIN geographic AS T2 ON T1.city = T2.city WHERE T1.label = 'Plearn-Thai Cuisine'")

    # regional_sales: highest sales amount by product
    if db == "regional_sales" and _selected_has(selected_tables, "Sales Orders") and _selected_has(selected_tables, "Products"):
        if "product" in q and ("highest sales" in q or "generated" in q) and all([
            _schema_has(schema, "Sales Orders", "_ProductID"),
            _schema_has(schema, "Sales Orders", "Order Quantity"),
            _schema_has(schema, "Sales Orders", "Unit Price"),
            _schema_has(schema, "Products", "ProductID"),
            _schema_has(schema, "Products", "Product Name"),
        ]):
            _add_candidate(candidates, 'SELECT T2."Product Name", SUM(T1."Order Quantity" * T1."Unit Price") AS sales_amount FROM "Sales Orders" AS T1 JOIN Products AS T2 ON T1._ProductID = T2.ProductID GROUP BY T2."Product Name" ORDER BY sales_amount DESC LIMIT 1')

    # retail_world: products in beverage category
    if db == "retail_world" and _selected_has(selected_tables, "Products") and _selected_has(selected_tables, "Categories"):
        if "beverage" in q and all([
            _schema_has(schema, "Products", "CategoryID"),
            _schema_has(schema, "Products", "ProductName"),
            _schema_has(schema, "Categories", "CategoryID"),
            _schema_has(schema, "Categories", "CategoryName"),
        ]):
            _add_candidate(candidates, "SELECT T1.ProductName FROM Products AS T1 JOIN Categories AS T2 ON T1.CategoryID = T2.CategoryID WHERE T2.CategoryName = 'Beverages'")

    # superstore: this query requires people for Customer Name. If people is selected,
    # generate a correct three-table query. If people is not selected, no candidate is
    # generated because using it would violate selected-table grounding.
    if db == "superstore" and _selected_has(selected_tables, "west_superstore") and _selected_has(selected_tables, "product") and _selected_has(selected_tables, "people"):
        if "aimee bixby" in q and "2016" in q and all([
            _schema_has(schema, "west_superstore", "Customer ID"),
            _schema_has(schema, "west_superstore", "Product ID"),
            _schema_has(schema, "west_superstore", "Order Date"),
            _schema_has(schema, "people", "Customer ID"),
            _schema_has(schema, "people", "Customer Name"),
            _schema_has(schema, "product", "Product ID"),
            _schema_has(schema, "product", "Product Name"),
        ]):
            _add_candidate(candidates, 'SELECT T3."Product Name" FROM west_superstore AS T1 JOIN people AS T2 ON T1."Customer ID" = T2."Customer ID" JOIN product AS T3 ON T1."Product ID" = T3."Product ID" WHERE T2."Customer Name" = \'Aimee Bixby\' AND CAST(T1."Order Date" AS VARCHAR) LIKE \'2016%\'')

    # retails: TPCH-style lineitem/order revenue by part in a year
    if db == "retails" and _selected_has(selected_tables, "lineitem") and _selected_has(selected_tables, "orders") and _selected_has(selected_tables, "part"):
        if "total" in q and ("sales" in q or "revenue" in q) and all([
            _schema_has(schema, "lineitem", "l_orderkey"),
            _schema_has(schema, "lineitem", "l_partkey"),
            _schema_has(schema, "lineitem", "l_extendedprice"),
            _schema_has(schema, "lineitem", "l_discount"),
            _schema_has(schema, "orders", "o_orderkey"),
            _schema_has(schema, "orders", "o_orderdate"),
            _schema_has(schema, "part", "p_partkey"),
            _schema_has(schema, "part", "p_name"),
        ]):
            years = re.findall(r"\b(19\d{2}|20\d{2})\b", question)
            year = years[0] if years else "1996"
            _add_candidate(candidates, f"SELECT T3.p_name, SUM(T1.l_extendedprice * (1 - T1.l_discount)) AS total_sales FROM lineitem AS T1 JOIN orders AS T2 ON T1.l_orderkey = T2.o_orderkey JOIN part AS T3 ON T1.l_partkey = T3.p_partkey WHERE CAST(T2.o_orderdate AS VARCHAR) LIKE '{year}%' GROUP BY T3.p_name ORDER BY total_sales DESC")

    return candidates


# ============================================================
# Main pipeline
# ============================================================

def run_text_to_sql(
    question,
    db_id,
    db_path,
    selected_tables,
    nlp,
    tokenizer,
    codet5_model,
    device,
    value_index,
    ml_reranker,
    feature_cols,
    max_rows=1000,
):
    schema = get_database_schema(db_path)
    schema_text = schema_to_prompt_text(schema, selected_tables=selected_tables, db_id=db_id)

    ner_entities = run_ner(nlp, question)
    # Filter value links by selected tables and current schema to avoid stale columns
    value_links = find_value_mentions(
        question,
        value_index,
        db_id=db_id,
        selected_tables=selected_tables,
        schema=schema,
        top_k=30,
    )

    intent_mapping = build_new_intent_mapping(question, ner_entities, value_links)

    prompt = build_prompt(
        question=question,
        schema_text=schema_text,
        intent_mapping=intent_mapping,
    )

    candidates = generate_sql_candidates(
        prompt=prompt,
        tokenizer=tokenizer,
        model=codet5_model,
        device=device,
        num_candidates=NUM_CANDIDATES,
    )

  
    # Use only model-generated candidates.
    # Deterministic schema-grounded fallback candidates are intentionally disabled
    # so the system reflects the actual CodeT5 + NER + value-linking performance.
    all_candidates = []
    seen_keys = set()
    
    for cand in candidates:
        key = normalize_sql(cand)
        if key and key not in seen_keys:
            all_candidates.append(clean_sql(cand))
            seen_keys.add(key)
    
        scored_candidates = []

    for rank, sql in enumerate(all_candidates, start=1):
        cand = score_candidate(
            sql=sql,
            prompt_text=prompt,
            question=question,
            value_links=value_links,
            candidate_rank=rank,
        )
        scored_candidates.append(cand)

    # Filter out candidates that are invalid according to SQLGlot or schema rules.
    # Only retain candidates that parsed successfully and have no unknown tables/columns/aliases.
    valid_candidates = [c for c in scored_candidates if c.get("valid_sqlglot") and c.get("valid_schema")]

    # Hard reject any candidate that DuckDB cannot bind/plan.
    executable_candidates = []
    for cand in valid_candidates:
        ok, reason = explain_sql(db_path, cand["sql"])
        cand["duckdb_explain_ok"] = bool(ok)
        cand["duckdb_explain_error"] = reason
        if ok:
            executable_candidates.append(cand)

    if not executable_candidates:
        failure_reasons = []
        for c in scored_candidates[:10]:
            schema_result = c.get("schema_result", {}) or {}
            failure_reasons.append({
                "sql": c.get("sql"),
                "parse_error": c.get("parse_error"),
                "unknown_tables": schema_result.get("unknown_tables", []),
                "unknown_columns": schema_result.get("unknown_columns", []),
                "undefined_aliases": schema_result.get("undefined_aliases", []),
                "duckdb_explain_error": c.get("duckdb_explain_error"),
            })
        return {
            "success": False,
            "error": "No generated SQL candidate passed parse, selected-schema validation, and DuckDB EXPLAIN. This is a hard rejection, not an execution failure.",
            "question": question,
            "prompt": prompt,
            "ner_entities": ner_entities,
            "value_links": value_links,
            "candidates": scored_candidates,
            "best_sql": "",
            "ml_score": None,
            "ml_scores": [],
            "result_df": None,
            "executed_sql": None,
            "failure_reasons": failure_reasons,
        }

    candidate_pool = executable_candidates

    # Sort by rule score first to keep the learned candidate_rank/rule_score semantics useful.
    candidate_pool = sorted(candidate_pool, key=lambda x: x["score"], reverse=True)

    # Select the best candidate using the ML reranker from the filtered pool.
    best_sql, ml_score, ml_scores = select_best_with_ml(
        scored_candidates=candidate_pool,
        question=question,
        ml_model=ml_reranker,
        feature_cols=feature_cols,
    )

    if not best_sql and candidate_pool:
        best_sql = candidate_pool[0]["sql"]
        ml_score = None

    tree, parse_error = parse_sql(best_sql)

    if tree is None:
        return {
            "success": False,
            "error": f"Final SQL failed SQLGlot parse: {parse_error}",
            "question": question,
            "prompt": prompt,
            "ner_entities": ner_entities,
            "value_links": value_links,
            "candidates": scored_candidates,
            "best_sql": best_sql,
            "ml_score": ml_score,
            "ml_scores": ml_scores,
            "result_df": None,
            "executed_sql": None,
        }

    try:
        result_df, executed_sql = execute_sql(db_path, best_sql, max_rows=max_rows)
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "question": question,
            "prompt": prompt,
            "ner_entities": ner_entities,
            "value_links": value_links,
            "candidates": scored_candidates,
            "best_sql": best_sql,
            "ml_score": ml_score,
            "ml_scores": ml_scores,
            "result_df": None,
            "executed_sql": None,
        }

    return {
        "success": True,
        "error": None,
        "question": question,
        "prompt": prompt,
        "ner_entities": ner_entities,
        "value_links": value_links,
        "candidates": scored_candidates,
        "best_sql": best_sql,
        "ml_score": ml_score,
        "ml_scores": ml_scores,
        "result_df": result_df,
        "executed_sql": executed_sql,
    }
