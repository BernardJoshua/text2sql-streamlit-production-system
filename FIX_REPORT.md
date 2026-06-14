# Fix Report for Text‑to‑SQL Production System

## Root Cause Analysis

The original Streamlit text‑to‑SQL application was trained on an older version of the
`retail_complains` schema which contained noisy or deprecated columns such as
`Consumer disputed?`, `Timely response?`, `Consumer consent provided?`, and a number
of synthetic fields like `issue_lower` and `consent`. When the underlying DuckDB
database was cleaned, those columns were removed; however several parts of the
production system still referenced the outdated schema:

* The **value index dataset** contained entries linking common phrases (e.g. “Yes”) to
  the removed columns. Because `find_value_mentions` only filtered by database ID and
  did not cross‑check the current schema or selected tables, these stale
  mappings polluted the intent mapping and misled the model.
* The **prompt schema extraction** was correct, but **candidate validation** merely
  penalized unknown tables, columns and aliases. This meant that the reranker
  could still choose a high‑scoring candidate even if it referenced an undefined
  alias (`T3`) or a removed column. The final SQL occasionally included
  expressions such as `events."Consumer disputed?"` or undefined aliases like
  `T3` which promptly failed at execution time.
* In the Streamlit UI the **value index** was cached via `st.cache_data`, so any
  corrections to the underlying value index would not propagate until the cache
  expired.

Together these factors caused the system to “hallucinate” columns that were no longer
present in DuckDB and to generate invalid SQL candidates. The bug was not fixed by
simply cleaning the DuckDB schema, because the value index and scoring logic still
carried stale state forward.

## Changes Made

The following updates were implemented to permanently resolve the above issues while
preserving the training‑compatible prompt format:

1. **Value index filtering** (`find_value_mentions` in `text2sql_engine.py`)
   * The function now accepts `selected_tables` and the current `schema` as
     arguments. When scanning the value index it filters out any row where the
     table is not among the user‑selected tables or the column does not exist in
     the actual DuckDB schema. This prevents values from removed or unselected
     columns (such as `Consumer disputed?`) from being linked to the question.
   * The fuzzy matching logic and deduplication remain unchanged, but the candidate
     set is guaranteed to be grounded in the current database structure.

2. **Invalid candidate rejection** (`score_candidate` and `run_text_to_sql`)
   * A new field `valid_schema` is included in the candidate metadata. It is set
     to `False` whenever `validate_schema_compatibility` detects unknown tables,
     unknown columns, or undefined aliases in the SQL AST.
   * Candidates that fail to parse via SQLGlot or that have `valid_schema=False` are
     removed from consideration prior to reranking. The system now filters
     `scored_candidates` down to those that both parse and satisfy the schema
     constraints. If no valid candidate remains, it falls back to the full set,
     but such cases will propagate a clear error message.

3. **Fresh value index loading** (`app.py`)
   * The Streamlit `cached_load_value_index` function has been stripped of its
     `@st.cache_data` decorator. Each run loads the value index anew, ensuring
     that schema updates on Hugging Face or local overrides are honoured and
     eliminating stale mappings from memory.

4. **Testing infrastructure** (`run_text2sql_regression_tests.py`)
   * A new regression test script exercises the entire pipeline across nine
     representative databases. It runs natural language queries in the
     BIRD‑style described in the prompt, captures the generated SQL, checks
     parsing and schema validity, executes the query in DuckDB, and stores a
     small preview of the result. The script writes both a CSV and a Markdown
     summary for easy inspection.
   * This script can be used during development and CI to prevent regressions
     where invalid aliases or stale columns reappear.

5. **Packaging hygiene**
   * Removed any use of Streamlit’s data caching for value index to avoid
     persisting stale data. Only model loading remains cached.
   * The patch ensures that `.cache` directories generated at runtime are
     excluded when repackaging.

## Remaining Limitations

* The CodeT5 model may occasionally hallucinate columns or table aliases that do
  not exist. With the stricter schema validation these candidates will be
  discarded, but if no candidate remains the system falls back to the highest
  scoring (invalid) candidate and the user will see an explicit error message.
* Execution still relies on the availability of the `duckdb` Python module. If
  DuckDB is not installed in the runtime environment the regression tests will
  be unable to execute queries; however, the schema and parsing validation will
  still function.
* The LightGBM reranker was trained on the original candidate distribution.
  Rejecting invalid candidates may alter the distribution slightly. In practice
  this had negligible impact on accuracy during testing, but retraining could
  further improve ranking.

## How to Run the App

1. Install dependencies (ensure `duckdb`, `sqlglot`, `spacy`, `transformers`, and
   `lightgbm` are available). A requirements file is included.
2. (Optional) Provide a Hugging Face token via the environment variable
   `HF_TOKEN` or `.streamlit/secrets.toml` if your models or value index are
   private.
3. Build the DuckDB databases by running:

   ```bash
   python scripts/build_duckdb_from_bundle.py
   ```

4. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

   Select a database and tables, enter a natural language question, and
   optionally view the prompt, candidates, and debug information.

## How to Run Regression Tests

Execute the provided regression test script to verify that the system
generates syntactically and semantically correct SQL across multiple
databases:

```bash
python run_text2sql_regression_tests.py --hf-token YOUR_HF_TOKEN --out-dir results
```

The script will output `regression_test_results.csv` and
`regression_test_results.md` in the specified directory. These files
list the generated SQL, whether it parsed and matched the schema, whether it
executed in DuckDB, a preview of the result, and any notes about
exceptions or errors.

## Additional Fix: Local Value Index Fallback

After deployment, the app failed with a Hugging Face `RepositoryNotFoundError` / `401 Client Error` when trying to download `BernardJoshua/spacy-ner-dataset/value_index.jsonl`. The corrected version now treats `data/value_index.jsonl` bundled inside the project as the primary value-index source. If this file is missing, the loader extracts it from `data/duckdb_text2sql_bundle.zip`. Hugging Face download is now only a fallback, so the app can start even when the dataset repository is private, renamed, or unavailable.

This means the included zip is self-contained for the value-index layer. Hugging Face authentication is still needed only for private model repositories.
