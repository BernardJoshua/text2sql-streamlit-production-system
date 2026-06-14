import os
from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    SPACY_NER_REPO_ID,
    CODET5_REPO_ID,
    LIGHTGBM_RERANKER_REPO_ID,
    VALUE_INDEX_REPO_ID,
    VALUE_INDEX_FILENAME,
    NUM_CANDIDATES,
    DB_DIR,
)

from text2sql_engine import (
    load_spacy_ner,
    load_codet5,
    load_lightgbm_reranker,
    load_value_index,
    list_duckdb_files,
    get_db_id_from_file,
    get_database_schema,
    run_text_to_sql,
    execute_sql,
)


st.set_page_config(
    page_title="Text-to-SQL Analytics Chatbot",
    page_icon="🦆",
    layout="wide",
)


def get_hf_token():
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token

    try:
        return st.secrets.get("HF_TOKEN", None)
    except Exception:
        return None


def format_sql_for_editor(sql: str) -> str:
    """
    Lightweight SQL formatter so generated SQL is easier to read/edit in Streamlit.
    Does not require extra packages.
    """
    if not sql:
        return ""

    sql = str(sql).strip().rstrip(";")

    replacements = [
        (" SELECT ", "\nSELECT "),
        (" FROM ", "\nFROM "),
        (" INNER JOIN ", "\nINNER JOIN "),
        (" LEFT JOIN ", "\nLEFT JOIN "),
        (" RIGHT JOIN ", "\nRIGHT JOIN "),
        (" FULL JOIN ", "\nFULL JOIN "),
        (" JOIN ", "\nJOIN "),
        (" ON ", "\n  ON "),
        (" WHERE ", "\nWHERE "),
        (" AND ", "\n  AND "),
        (" OR ", "\n  OR "),
        (" GROUP BY ", "\nGROUP BY "),
        (" ORDER BY ", "\nORDER BY "),
        (" HAVING ", "\nHAVING "),
        (" LIMIT ", "\nLIMIT "),
    ]

    formatted = " " + sql + " "
    for old, new in replacements:
        formatted = formatted.replace(old, new)

    formatted = formatted.strip()
    formatted = "\n".join(
        line.strip()
        for line in formatted.splitlines()
        if line.strip()
    )

    return formatted + ";"


def reset_manual_failure_console():
    """
    Reset only manual-console UI state.
    Does not cache query data/results.
    """
    st.session_state["show_manual_console_after_failure"] = False
    st.session_state["last_generation_error"] = None
    st.session_state["manual_console_nonce"] = st.session_state.get("manual_console_nonce", 0) + 1


@st.cache_resource(show_spinner="Loading spaCy NER model from Hugging Face...")
def cached_load_spacy_ner(hf_token):
    return load_spacy_ner(hf_token=hf_token)


@st.cache_resource(show_spinner="Loading CodeT5 model from Hugging Face...")
def cached_load_codet5(hf_token):
    return load_codet5(hf_token=hf_token)


@st.cache_resource(show_spinner="Loading LightGBM reranker from Hugging Face...")
def cached_load_lightgbm(hf_token):
    return load_lightgbm_reranker(hf_token=hf_token)


def cached_load_value_index(hf_token):
    """
    Value index is intentionally not Streamlit-cached.
    This prevents stale schema/table/value mappings from leaking into new runs.
    """
    return load_value_index(hf_token=hf_token)


def render_manual_sql_console_after_failure(selected_db_path, selected_db_label, max_rows):
    """
    Manual SQL console appears only after generation failure.
    It starts empty for each new generation failure.
    It does not store result dataframes in session_state.
    """
    nonce = st.session_state.get("manual_console_nonce", 0)
    manual_key = f"manual_sql_console_after_failure_{nonce}"

    st.warning("Query generation failed. You can manually run SQL below.")

    with st.expander("Manual SQL Console", expanded=True):
        st.caption(
            "This appears only because the generated query failed validation. "
            "Manual SQL runs directly against the selected DuckDB database."
        )

        with st.form("manual_sql_after_failure_form"):
            manual_sql = st.text_area(
                "Write or paste SQL here",
                height=260,
                key=manual_key,
                placeholder="""SELECT *
FROM your_table
LIMIT 100;""",
            )

            st.caption(f"Running against: `{selected_db_label}`")

            col1, col2 = st.columns([1, 1])

            with col1:
                run_manual_sql = st.form_submit_button("Execute Manual SQL", type="primary")

            with col2:
                clear_manual_sql = st.form_submit_button("Clear SQL Console")

        if clear_manual_sql:
            st.session_state[manual_key] = ""
            st.rerun()

        if run_manual_sql:
            if not manual_sql.strip():
                st.warning("Manual SQL console is empty.")
            else:
                try:
                    manual_result_df, manual_executed_sql = execute_sql(
                        db_path=selected_db_path,
                        sql=manual_sql,
                        max_rows=max_rows,
                    )

                    st.success("Manual SQL executed successfully.")

                    with st.expander("Executed Manual SQL", expanded=False):
                        st.text_area(
                            "Executed SQL",
                            value=format_sql_for_editor(manual_executed_sql),
                            height=220,
                            disabled=True,
                        )

                    st.subheader("Manual SQL Result")
                    st.dataframe(manual_result_df, use_container_width=True)

                    manual_csv = manual_result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download manual SQL result CSV",
                        data=manual_csv,
                        file_name="manual_query_result.csv",
                        mime="text/csv",
                    )

                except Exception as e:
                    st.error(f"Manual SQL execution failed: {e}")


# ============================================================
# Initial session flags
# ============================================================

if "show_manual_console_after_failure" not in st.session_state:
    st.session_state["show_manual_console_after_failure"] = False

if "last_generation_error" not in st.session_state:
    st.session_state["last_generation_error"] = None

if "manual_console_nonce" not in st.session_state:
    st.session_state["manual_console_nonce"] = 0


# ============================================================
# App header
# ============================================================

st.title("Text-to-SQL Analytics Chatbot")
st.caption("spaCy NER + value index + CodeT5 + LightGBM reranker + DuckDB execution")

hf_token = get_hf_token()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.header("Models loaded")

    st.code(
        f"""spaCy NER:
{SPACY_NER_REPO_ID}

CodeT5:
{CODET5_REPO_ID}

LightGBM reranker:
{LIGHTGBM_RERANKER_REPO_ID}

Value index:
{VALUE_INDEX_REPO_ID}/{VALUE_INDEX_FILENAME}

Candidates:
{NUM_CANDIDATES}""",
        language="text",
        wrap_lines=True,
    )

    if hf_token:
        st.success("HF_TOKEN detected.")
    else:
        st.warning("No HF_TOKEN detected. Public repos will work. Private repos need .streamlit/secrets.toml.")

    st.header("Database")

    db_files = list_duckdb_files()

    if not db_files:
        st.error(
            f"No DuckDB files found in:\n{DB_DIR}\n\n"
            "Run: python scripts/build_duckdb_from_bundle.py"
        )
        st.stop()

    db_labels = [p.name for p in db_files]
    selected_db_label = st.selectbox("Select database", db_labels)
    selected_db_path = db_files[db_labels.index(selected_db_label)]
    selected_db_id = get_db_id_from_file(selected_db_path)

    st.write("Detected db_id")
    st.code(selected_db_id, language="text", wrap_lines=True)

    schema = get_database_schema(selected_db_path)
    table_names = list(schema.keys())

    selected_tables = st.multiselect(
        "Tables to include in prompt",
        table_names,
        default=table_names,
    )

    max_rows = st.number_input(
        "Max rows to return",
        min_value=10,
        max_value=10000,
        value=1000,
        step=10,
    )

    st.header("Debug")
    show_prompt = st.checkbox("Show exact prompt", value=False)
    show_candidates = st.checkbox("Show candidate SQLs", value=True)
    show_debug = st.checkbox("Show NER/value links", value=True)


# ============================================================
# Load resources
# ============================================================

nlp = cached_load_spacy_ner(hf_token)
tokenizer, codet5_model, device = cached_load_codet5(hf_token)
ml_reranker, feature_cols = cached_load_lightgbm(hf_token)
value_index = cached_load_value_index(hf_token)

st.success("All models/resources loaded.")


# ============================================================
# Question input
# ============================================================

st.subheader("Ask a question")

question = st.text_area(
    "Natural language question",
    placeholder="Example: In 2015, how many complaints about Billing disputes were sent by clients in Portland?",
    height=100,
)

run_button = st.button("Generate SQL and Execute", type="primary")


# ============================================================
# Query generation
# ============================================================

if run_button:
    if not question.strip():
        st.warning("Enter a question first.")
        st.stop()

    if not selected_tables:
        st.warning("Select at least one table.")
        st.stop()

    with st.spinner("Generating SQL candidates, reranking, and executing DuckDB query..."):
        output = run_text_to_sql(
            question=question,
            db_id=selected_db_id,
            db_path=selected_db_path,
            selected_tables=selected_tables,
            nlp=nlp,
            tokenizer=tokenizer,
            codet5_model=codet5_model,
            device=device,
            value_index=value_index,
            ml_reranker=ml_reranker,
            feature_cols=feature_cols,
            max_rows=max_rows,
        )

    if output["success"]:
        reset_manual_failure_console()

        st.subheader("Selected SQL")
        st.text_area(
            "Selected SQL",
            value=format_sql_for_editor(output["best_sql"]),
            height=220,
            disabled=True,
        )

        if output["ml_score"] is not None:
            st.metric("LightGBM reranker score", round(float(output["ml_score"]), 4))

        with st.expander("Executed SQL", expanded=False):
            st.text_area(
                "Executed SQL",
                value=format_sql_for_editor(output["executed_sql"]),
                height=220,
                disabled=True,
            )

        st.subheader("Result")
        st.dataframe(output["result_df"], use_container_width=True)

        csv = output["result_df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download result CSV",
            data=csv,
            file_name="query_result.csv",
            mime="text/csv",
        )

    else:
        st.session_state["show_manual_console_after_failure"] = True
        st.session_state["last_generation_error"] = output["error"]
        st.session_state["manual_console_nonce"] = st.session_state.get("manual_console_nonce", 0) + 1

        st.error(output["error"])

        render_manual_sql_console_after_failure(
            selected_db_path=selected_db_path,
            selected_db_label=selected_db_label,
            max_rows=max_rows,
        )

    if show_debug:
        st.subheader("NER entities")
        st.json(output["ner_entities"])

        st.subheader("Value index links")
        if output["value_links"]:
            st.dataframe(pd.DataFrame(output["value_links"]), use_container_width=True)
        else:
            st.info("No linked database values found.")

    if show_candidates:
        st.subheader("Top candidate SQLs")

        rows = []
        for c in output["candidates"][:15]:
            rows.append({
                "candidate_rank": c.get("candidate_rank"),
                "rule_score": c.get("score"),
                "valid_sqlglot": c.get("valid_sqlglot"),
                "valid_schema": c.get("valid_schema"),
                "sql": format_sql_for_editor(c.get("sql", "")),
                "reasons": " | ".join(c.get("reasons", [])[:8]),
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if show_prompt:
        st.subheader("Exact prompt sent to CodeT5")
        st.text(output["prompt"])


# ============================================================
# Manual console persists only after failure, not always
# ============================================================

elif st.session_state.get("show_manual_console_after_failure"):
    if st.session_state.get("last_generation_error"):
        st.error(st.session_state["last_generation_error"])

    render_manual_sql_console_after_failure(
        selected_db_path=selected_db_path,
        selected_db_label=selected_db_label,
        max_rows=max_rows,
    )