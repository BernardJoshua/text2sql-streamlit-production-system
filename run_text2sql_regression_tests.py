"""
Regression testing script for the Text-to-SQL Streamlit system.

This script programmatically exercises the text-to-SQL engine across a set of
databases, selected tables, and natural language questions. It loads the
production models (spaCy NER, CodeT5, LightGBM reranker, and value index)
exactly as the Streamlit app does. For each test case the script invokes
`run_text_to_sql` and captures whether the generated SQL parses, passes
schema validation, executes in DuckDB, and what the result looks like.

The output is written to two files:

* ``regression_test_results.csv`` – machine‑readable summary of test outcomes.
* ``regression_test_results.md`` – human‑readable table with key details.

Example usage::

    python run_text2sql_regression_tests.py --hf-token $HF_TOKEN

If no Hugging Face token is provided the script will attempt to load
public resources. The databases are expected to live in
``text2sql_streamlit_production_system/duckdb_databases`` relative to the
script location.

Note: Running this script can take several minutes because it loads
the CodeT5 model and generates multiple SQL candidates for each query.
"""

import argparse
import json
import csv
from pathlib import Path

import pandas as pd

from text2sql_engine import (
    load_spacy_ner,
    load_codet5,
    load_lightgbm_reranker,
    load_value_index,
    run_text_to_sql,
    get_database_schema,
    schema_to_prompt_text,
    parse_sql,
    validate_schema_compatibility,
    get_db_id_from_file,
)
from config import DB_DIR


def run_tests(test_cases, hf_token=None, max_rows=10):
    """Run a suite of tests and return results as a list of dictionaries."""

    # Load models and resources once
    nlp = load_spacy_ner(hf_token=hf_token)
    tokenizer, codet5_model, device = load_codet5(hf_token=hf_token)
    ml_reranker, feature_cols = load_lightgbm_reranker(hf_token=hf_token)
    value_index = load_value_index(hf_token=hf_token)

    results = []

    for case in test_cases:
        db_label = case["db"]
        tables = case["tables"]
        question = case["question"]

        # Locate database file
        db_path = None
        for f in Path(DB_DIR).glob("*.duckdb"):
            if f.stem.startswith(db_label):
                db_path = f
                break
        if not db_path:
            results.append({
                "db": db_label,
                "tables": ",".join(tables),
                "question": question,
                "generated_sql": "",
                "parsed": False,
                "valid_schema": False,
                "executed": False,
                "result_preview": "Database not found",
                "notes": f"Database {db_label} not found in {DB_DIR}",
            })
            continue

        # Run the text‑to‑SQL pipeline
        try:
            output = run_text_to_sql(
                question=question,
                db_id=get_db_id_from_file(db_path),
                db_path=db_path,
                selected_tables=tables,
                nlp=nlp,
                tokenizer=tokenizer,
                codet5_model=codet5_model,
                device=device,
                value_index=value_index,
                ml_reranker=ml_reranker,
                feature_cols=feature_cols,
                max_rows=max_rows,
            )
        except Exception as e:
            results.append({
                "db": db_label,
                "tables": ",".join(tables),
                "question": question,
                "generated_sql": "",
                "parsed": False,
                "valid_schema": False,
                "executed": False,
                "result_preview": "",
                "notes": f"Exception during run_text_to_sql: {e}",
            })
            continue

        best_sql = output.get("best_sql", "")
        executed = output.get("success", False)
        parse_ok = False
        schema_ok = False
        result_preview = ""
        notes = ""

        # Determine parse and schema validity by reusing validation helpers
        if best_sql:
            tree, parse_err = parse_sql(best_sql)
            parse_ok = tree is not None and parse_err is None
            # Build prompt schema text for selected tables only
            schema = get_database_schema(db_path)
            schema_text = schema_to_prompt_text(schema, selected_tables=tables, db_id=get_db_id_from_file(db_path))
            prompt_text = f"Relevant Table Schemas:\n{schema_text}\nIntent Mapping:"
            if tree is not None:
                schema_res = validate_schema_compatibility(tree, prompt_text)
                schema_ok = not (
                    schema_res.get("unknown_tables")
                    or schema_res.get("unknown_columns")
                    or schema_res.get("undefined_aliases")
                )
            else:
                schema_ok = False

        if executed and output.get("result_df") is not None:
            df = output["result_df"]
            preview_df = df.head(3)
            # Convert preview to a compact string
            result_preview = preview_df.to_csv(index=False).strip()
        else:
            if not executed:
                notes = output.get("error", "")

        results.append({
            "db": db_label,
            "tables": ",".join(tables),
            "question": question,
            "generated_sql": best_sql,
            "parsed": parse_ok,
            "valid_schema": schema_ok,
            "executed": bool(executed),
            "result_preview": result_preview,
            "notes": notes,
        })

    return results


def save_results(results, output_dir):
    """Write CSV and Markdown summaries to the specified directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "regression_test_results.csv"
    md_path = output_dir / "regression_test_results.md"

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "db", "tables", "question", "generated_sql", "parsed",
            "valid_schema", "executed", "result_preview", "notes",
        ])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # Write Markdown summary
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Regression Test Results\n\n")
        f.write("| DB | Tables | Question | Parsed | Valid Schema | Executed | Result Preview | Notes |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for row in results:
            preview = row["result_preview"].replace("\n", "<br>") if row["result_preview"] else ""
            notes = row["notes"].replace("\n", "<br>") if row["notes"] else ""
            f.write(
                f"| {row['db']} | {row['tables']} | {row['question']} | "
                f"{row['parsed']} | {row['valid_schema']} | {row['executed']} | "
                f"{preview} | {notes} |\n"
            )

    return csv_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Run text‑to‑SQL regression tests.")
    parser.add_argument(
        "--hf-token", dest="hf_token", default=None,
        help="Hugging Face token for accessing private repositories.",
    )
    parser.add_argument(
        "--out-dir", dest="out_dir", default=".",
        help="Directory to write regression_test_results.csv and .md",
    )
    args = parser.parse_args()

    # Define regression tests (natural BIRD‑style questions)
    test_cases = [
        {
            "db": "retail_complains",
            "tables": ["client", "events"],
            "question": "Among clients from Portland, how many complaints were about Billing disputes?",
        },
        {
            "db": "car_retails",
            "tables": ["products", "orderdetails"],
            "question": "Calculate the total quantity ordered for 18th Century Vintage Horse Carriage.",
        },
        {
            "db": "cars",
            "tables": ["data", "price"],
            "question": "How many cylinders does the cheapest car have?",
        },
        {
            "db": "sales",
            "tables": ["Sales", "Employees"],
            "question": "Which salesperson sold the most items in total?",
        },
        {
            "db": "restaurant",
            "tables": ["generalinfo", "geographic"],
            "question": "What is the county and region of Plearn-Thai Cuisine restaurant?",
        },
        {
            "db": "regional_sales",
            "tables": ["Sales Orders", "Products"],
            "question": "Which product generated the highest sales amount?",
        },
        {
            "db": "retail_world",
            "tables": ["Products", "Categories"],
            "question": "What are the products that belong to the beverage category?",
        },
        {
            "db": "superstore",
            # Aimee Bixby is stored in people.Customer Name. Without people,
            # this question is impossible under selected-table grounding.
            "tables": ["west_superstore", "people", "product"],
            "question": "Please list the products ordered by Aimee Bixby in 2016.",
        },
        {
            "db": "retails",
            "tables": ["lineitem", "orders", "part"],
            "question": "List the total sales amount for each product in 1996.",
        },
    ]

    results = run_tests(test_cases, hf_token=args.hf_token)
    csv_path, md_path = save_results(results, args.out_dir)

    print(f"Wrote results to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()