# Text-to-SQL Streamlit Production System

This is the production version of your Text-to-SQL system.

It loads:

- spaCy NER model: `BernardJoshua/text-to-sql-spacy-ner-final`
- CodeT5 model: `BernardJoshua/codet5-small-text-to-sql-prompt-final_model`
- LightGBM SQL reranker: `BernardJoshua/text-to-sql-lightgbm-reranker`
- Value index dataset: `BernardJoshua/spacy-ner-dataset/value_index.jsonl`
- Local DuckDB databases generated from `data/duckdb_text2sql_bundle.zip`

## Production pipeline

```text
User question
-> spaCy NER extracts spans
-> value_index.jsonl links values to table.column.value
-> prompt is built in the SAME outer format used during CodeT5 training
-> CodeT5 generates 30 SQL candidates
-> rule scorer computes SQLGlot/schema/value-index features
-> LightGBM reranker selects best SQL
-> SQL is validated as SELECT-only
-> DuckDB executes SQL
-> Streamlit displays result
```

## Important prompt format

The prompt format is intentionally preserved:

```text
Question:
...

Relevant Table Schemas:
...

Intent Mapping:
...

SQL generation rules:
...
```

Only the content inside `Intent Mapping:` is produced dynamically at inference time.

This matches the training notebook design, where the outer prompt format was not changed and only the new mapping content replaced the old inaccurate mapping.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your Hugging Face token

If your repos are private, create:

```text
.streamlit/secrets.toml
```

with:

```toml
HF_TOKEN = "hf_your_token_here"
```

Or export it:

```bash
export HF_TOKEN="hf_your_token_here"
```

### 3. Build DuckDB files

The package includes:

```text
data/duckdb_text2sql_bundle.zip
```

Build `.duckdb` files:

```bash
python scripts/build_duckdb_from_bundle.py
```

This writes databases into:

```text
duckdb_databases/
```

### 4. Run Streamlit

```bash
streamlit run app.py
```

## Files

```text
app.py                              Streamlit dashboard
text2sql_engine.py                  Full production inference engine
config.py                           Model/resource config
requirements.txt                    Python dependencies
scripts/build_duckdb_from_bundle.py Builds .duckdb files from SQL bundle
scripts/upload_value_index_to_hf.py Optional value index upload helper
data/duckdb_text2sql_bundle.zip     Seed SQL + value index bundle
duckdb_databases/                   Generated DuckDB files go here
.streamlit/secrets.toml.example     Example HF token config
```

## Safety

The app only executes SQL that starts with `SELECT` and blocks destructive operations such as `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `CREATE`, `COPY`, and `PRAGMA`.


## Value index loading

This fixed build includes `data/value_index.jsonl` and loads it locally first. The app no longer requires `BernardJoshua/spacy-ner-dataset` to be publicly accessible just to start. Hugging Face is used only as a fallback when the local file and bundled zip are missing.

If the CodeT5, spaCy NER, or LightGBM model repositories are private, `HF_TOKEN` is still required for those model downloads.
