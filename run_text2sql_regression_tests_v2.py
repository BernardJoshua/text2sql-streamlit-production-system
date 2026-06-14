#!/usr/bin/env python3
"""
Regression test harness for the Streamlit CodeT5 Text-to-SQL system.

What it tests:
- single-table SQL
- join SQL
- COUNT
- SUM
- AVG
- MAX / ORDER BY LIMIT
- schema validity
- DuckDB execution
- result accuracy by comparing generated SQL output to golden expected SQL output

Run from the project root:

    python run_text2sql_regression_tests.py

Optional:

    python run_text2sql_regression_tests.py --max-cases 20
    python run_text2sql_regression_tests.py --no-ml
    python run_text2sql_regression_tests.py --clean-runtime-cache
    python run_text2sql_regression_tests.py --out regression_test_results.csv

Accuracy definition:
- execution_success_rate = generated SQL executed in DuckDB
- result_accuracy = generated SQL result exactly matches the expected SQL result
- schema_valid_rate = generated SQL passed selected-table schema validation
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    import duckdb
except Exception as e:
    raise RuntimeError("duckdb is required. Install with: pip install duckdb") from e


# ============================================================
# Test cases
# ============================================================

@dataclass
class TestCase:
    test_id: str
    db_id: str
    selected_tables: List[str]
    question: str
    expected_sql: str
    test_type: str          # single / join
    operation: str          # count / sum / avg / max / lookup / groupby
    difficulty: str = "normal"


TEST_CASES: List[TestCase] = [
    # ========================================================
    # retail_complains
    # ========================================================
    TestCase(
        test_id="retail_complaints_single_count_issue",
        db_id="retail_complains",
        selected_tables=["events"],
        question="How many complaints were about Billing disputes?",
        expected_sql='SELECT COUNT("Complaint ID") FROM events WHERE Issue = \'Billing disputes\';',
        test_type="single",
        operation="count",
    ),
    TestCase(
        test_id="retail_complaints_single_count_date",
        db_id="retail_complains",
        selected_tables=["events"],
        question="How many complaints were received in 2015?",
        expected_sql='SELECT COUNT("Complaint ID") FROM events WHERE "Date received" LIKE \'2015%\';',
        test_type="single",
        operation="count",
    ),
    TestCase(
        test_id="retail_complaints_single_top_issue",
        db_id="retail_complains",
        selected_tables=["events"],
        question="Which complaint issue appears most often?",
        expected_sql='SELECT Issue FROM events GROUP BY Issue ORDER BY COUNT("Complaint ID") DESC LIMIT 1;',
        test_type="single",
        operation="max",
    ),
    TestCase(
        test_id="retail_complaints_join_count_city_issue",
        db_id="retail_complains",
        selected_tables=["client", "events"],
        question="Among clients from Portland, how many complaints were about Billing disputes?",
        expected_sql=(
            'SELECT COUNT(T2."Complaint ID") '
            'FROM client AS T1 '
            'INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID '
            "WHERE T1.city = 'Portland' AND T2.Issue = 'Billing disputes';"
        ),
        test_type="join",
        operation="count",
    ),
    TestCase(
        test_id="retail_complaints_join_top_city",
        db_id="retail_complains",
        selected_tables=["client", "events"],
        question="Which city had the most complaints?",
        expected_sql=(
            'SELECT T1.city '
            'FROM client AS T1 '
            'INNER JOIN events AS T2 ON T1.client_id = T2.Client_ID '
            'GROUP BY T1.city '
            'ORDER BY COUNT(T2."Complaint ID") DESC '
            'LIMIT 1;'
        ),
        test_type="join",
        operation="max",
    ),

    # ========================================================
    # car_retails
    # ========================================================
    TestCase(
        test_id="car_retails_single_count_productline",
        db_id="car_retails",
        selected_tables=["products"],
        question="How many products are in the Classic Cars product line?",
        expected_sql="SELECT COUNT(productCode) FROM products WHERE productLine = 'Classic Cars';",
        test_type="single",
        operation="count",
    ),
    TestCase(
        test_id="car_retails_single_avg_msrp",
        db_id="car_retails",
        selected_tables=["products"],
        question="What is the average MSRP of all products?",
        expected_sql="SELECT AVG(MSRP) FROM products;",
        test_type="single",
        operation="avg",
    ),
    TestCase(
        test_id="car_retails_single_max_buyprice",
        db_id="car_retails",
        selected_tables=["products"],
        question="Which product has the highest buy price?",
        expected_sql="SELECT productName FROM products ORDER BY buyPrice DESC LIMIT 1;",
        test_type="single",
        operation="max",
    ),
    TestCase(
        test_id="car_retails_join_sum_quantity_product",
        db_id="car_retails",
        selected_tables=["products", "orderdetails"],
        question="Calculate the total quantity ordered for 18th Century Vintage Horse Carriage.",
        expected_sql=(
            "SELECT SUM(T2.quantityOrdered) "
            "FROM products AS T1 "
            "INNER JOIN orderdetails AS T2 ON T1.productCode = T2.productCode "
            "WHERE T1.productName = '18th Century Vintage Horse Carriage';"
        ),
        test_type="join",
        operation="sum",
    ),
    TestCase(
        test_id="car_retails_join_top_product_quantity",
        db_id="car_retails",
        selected_tables=["products", "orderdetails"],
        question="Which product was ordered the most?",
        expected_sql=(
            "SELECT T1.productName "
            "FROM products AS T1 "
            "INNER JOIN orderdetails AS T2 ON T1.productCode = T2.productCode "
            "GROUP BY T1.productName "
            "ORDER BY SUM(T2.quantityOrdered) DESC "
            "LIMIT 1;"
        ),
        test_type="join",
        operation="sum",
    ),
    TestCase(
        test_id="car_retails_join_top_customer_payment",
        db_id="car_retails",
        selected_tables=["customers", "payments"],
        question="Which customer made the highest total payment?",
        expected_sql=(
            "SELECT T1.customerName "
            "FROM customers AS T1 "
            "INNER JOIN payments AS T2 ON T1.customerNumber = T2.customerNumber "
            "GROUP BY T1.customerName "
            "ORDER BY SUM(T2.amount) DESC "
            "LIMIT 1;"
        ),
        test_type="join",
        operation="sum",
    ),

    # ========================================================
    # cars
    # ========================================================
    TestCase(
        test_id="cars_single_count_cylinders",
        db_id="cars",
        selected_tables=["data"],
        question="How many cars have 8 cylinders?",
        expected_sql="SELECT COUNT(ID) FROM data WHERE cylinders = 8;",
        test_type="single",
        operation="count",
    ),
    TestCase(
        test_id="cars_single_avg_mpg",
        db_id="cars",
        selected_tables=["data"],
        question="What is the average mpg of all cars?",
        expected_sql="SELECT AVG(mpg) FROM data;",
        test_type="single",
        operation="avg",
    ),
    TestCase(
        test_id="cars_single_max_horsepower",
        db_id="cars",
        selected_tables=["data"],
        question="Which car has the highest horsepower?",
        expected_sql="SELECT car_name FROM data ORDER BY horsepower DESC LIMIT 1;",
        test_type="single",
        operation="max",
    ),
    TestCase(
        test_id="cars_join_cheapest_cylinders",
        db_id="cars",
        selected_tables=["data", "price"],
        question="How many cylinders does the cheapest car have?",
        expected_sql=(
            "SELECT T1.cylinders "
            "FROM data AS T1 "
            "INNER JOIN price AS T2 ON T1.ID = T2.ID "
            "ORDER BY T2.price ASC "
            "LIMIT 1;"
        ),
        test_type="join",
        operation="max",
    ),
    TestCase(
        test_id="cars_join_country_chevy",
        db_id="cars",
        selected_tables=["data", "production"],
        question="Which country does Chevy C20 come from?",
        expected_sql=(
            "SELECT T2.country "
            "FROM data AS T1 "
            "INNER JOIN production AS T2 ON T1.ID = T2.ID "
            "WHERE lower(T1.car_name) = 'chevy c20';"
        ),
        test_type="join",
        operation="lookup",
    ),
    TestCase(
        test_id="cars_join_avg_price_8_cyl",
        db_id="cars",
        selected_tables=["data", "price"],
        question="What is the average price of cars with 8 cylinders?",
        expected_sql=(
            "SELECT AVG(T2.price) "
            "FROM data AS T1 "
            "INNER JOIN price AS T2 ON T1.ID = T2.ID "
            "WHERE T1.cylinders = 8;"
        ),
        test_type="join",
        operation="avg",
    ),

    # ========================================================
    # sales
    # ========================================================
    TestCase(
        test_id="sales_single_sum_quantity",
        db_id="sales",
        selected_tables=["Sales"],
        question="What is the total quantity sold?",
        expected_sql="SELECT SUM(Quantity) FROM Sales;",
        test_type="single",
        operation="sum",
    ),
    TestCase(
        test_id="sales_single_avg_quantity",
        db_id="sales",
        selected_tables=["Sales"],
        question="What is the average quantity sold?",
        expected_sql="SELECT AVG(Quantity) FROM Sales;",
        test_type="single",
        operation="avg",
    ),
    TestCase(
        test_id="sales_single_top_salesperson_id",
        db_id="sales",
        selected_tables=["Sales"],
        question="Which salesperson ID sold the highest total quantity?",
        expected_sql=(
            "SELECT SalesPersonID "
            "FROM Sales "
            "GROUP BY SalesPersonID "
            "ORDER BY SUM(Quantity) DESC "
            "LIMIT 1;"
        ),
        test_type="single",
        operation="sum",
    ),
    TestCase(
        test_id="sales_join_top_salesperson",
        db_id="sales",
        selected_tables=["Sales", "Employees"],
        question="Which salesperson sold the most items in total?",
        expected_sql=(
            "SELECT T2.FirstName, T2.LastName "
            "FROM Sales AS T1 "
            "INNER JOIN Employees AS T2 ON T1.SalesPersonID = T2.EmployeeID "
            "GROUP BY T2.FirstName, T2.LastName "
            "ORDER BY SUM(T1.Quantity) DESC "
            "LIMIT 1;"
        ),
        test_type="join",
        operation="sum",
    ),
    TestCase(
        test_id="sales_join_top_product_quantity",
        db_id="sales",
        selected_tables=["Sales", "Products"],
        question="Name the product that sold the highest quantity.",
        expected_sql=(
            "SELECT T2.Name "
            "FROM Sales AS T1 "
            "INNER JOIN Products AS T2 ON T1.ProductID = T2.ProductID "
            "GROUP BY T2.Name "
            "ORDER BY SUM(T1.Quantity) DESC "
            "LIMIT 1;"
        ),
        test_type="join",
        operation="sum",
    ),

    # ========================================================
    # restaurant
    # ========================================================
    TestCase(
        test_id="restaurant_single_count_japanese",
        db_id="restaurant",
        selected_tables=["generalinfo"],
        question="How many restaurants serve Japanese food?",
        expected_sql="SELECT COUNT(id_restaurant) FROM generalinfo WHERE food_type = 'japanese';",
        test_type="single",
        operation="count",
    ),
    TestCase(
        test_id="restaurant_single_food_type",
        db_id="restaurant",
        selected_tables=["generalinfo"],
        question="What food type does Plearn-Thai Cuisine serve?",
        expected_sql="SELECT food_type FROM generalinfo WHERE lower(label) = 'plearn-thai cuisine';",
        test_type="single",
        operation="lookup",
    ),
    TestCase(
        test_id="restaurant_join_county_region",
        db_id="restaurant",
        selected_tables=["generalinfo", "geographic"],
        question="What is the county and region of Plearn-Thai Cuisine restaurant?",
        expected_sql=(
            "SELECT T2.County, T2.region "
            "FROM generalinfo AS T1 "
            "INNER JOIN geographic AS T2 ON T1.city_lower = T2.city_lower "
            "WHERE lower(T1.label) = 'plearn-thai cuisine';"
        ),
        test_type="join",
        operation="lookup",
    ),
    TestCase(
        test_id="restaurant_join_street",
        db_id="restaurant",
        selected_tables=["generalinfo", "location"],
        question="What is the street name of Plearn-Thai Cuisine restaurant?",
        expected_sql=(
            "SELECT T2.street_name "
            "FROM generalinfo AS T1 "
            "INNER JOIN location AS T2 ON T1.id_restaurant = T2.id_restaurant "
            "WHERE lower(T1.label) = 'plearn-thai cuisine';"
        ),
        test_type="join",
        operation="lookup",
    ),

    # ========================================================
    # regional_sales
    # ========================================================
    TestCase(
        test_id="regional_sales_single_sum_sales_amount",
        db_id="regional_sales",
        selected_tables=["Sales Orders"],
        question="What is the total sales amount?",
        expected_sql='SELECT SUM("Order Quantity" * "Unit Price") FROM "Sales Orders";',
        test_type="single",
        operation="sum",
    ),
    TestCase(
        test_id="regional_sales_single_avg_discount",
        db_id="regional_sales",
        selected_tables=["Sales Orders"],
        question="What is the average discount applied?",
        expected_sql='SELECT AVG("Discount Applied") FROM "Sales Orders";',
        test_type="single",
        operation="avg",
    ),
    TestCase(
        test_id="regional_sales_single_top_product_id",
        db_id="regional_sales",
        selected_tables=["Sales Orders"],
        question="Which product ID generated the highest sales amount?",
        expected_sql=(
            'SELECT "_ProductID" '
            'FROM "Sales Orders" '
            'GROUP BY "_ProductID" '
            'ORDER BY SUM("Order Quantity" * "Unit Price") DESC '
            'LIMIT 1;'
        ),
        test_type="single",
        operation="sum",
    ),
    TestCase(
        test_id="regional_sales_join_top_product",
        db_id="regional_sales",
        selected_tables=["Sales Orders", "Products"],
        question="Which product generated the highest sales amount?",
        expected_sql=(
            'SELECT T2."Product Name" '
            'FROM "Sales Orders" AS T1 '
            'INNER JOIN Products AS T2 ON T1."_ProductID" = T2.ProductID '
            'GROUP BY T2."Product Name" '
            'ORDER BY SUM(T1."Order Quantity" * T1."Unit Price") DESC '
            'LIMIT 1;'
        ),
        test_type="join",
        operation="sum",
    ),

    # ========================================================
    # retail_world
    # ========================================================
    TestCase(
        test_id="retail_world_single_top_unit_price",
        db_id="retail_world",
        selected_tables=["Products"],
        question="Which product has the highest unit price?",
        expected_sql="SELECT ProductName FROM Products ORDER BY UnitPrice DESC LIMIT 1;",
        test_type="single",
        operation="max",
    ),
    TestCase(
        test_id="retail_world_single_avg_unit_price",
        db_id="retail_world",
        selected_tables=["Products"],
        question="What is the average unit price of products?",
        expected_sql="SELECT AVG(UnitPrice) FROM Products;",
        test_type="single",
        operation="avg",
    ),
    TestCase(
        test_id="retail_world_single_count_germany_customers",
        db_id="retail_world",
        selected_tables=["Customers"],
        question="How many customers are from Germany?",
        expected_sql="SELECT COUNT(CustomerID) FROM Customers WHERE Country = 'Germany';",
        test_type="single",
        operation="count",
    ),
    TestCase(
        test_id="retail_world_join_beverages",
        db_id="retail_world",
        selected_tables=["Products", "Categories"],
        question="What are the products that belong to the beverage category?",
        expected_sql=(
            "SELECT T1.ProductName "
            "FROM Products AS T1 "
            "INNER JOIN Categories AS T2 ON T1.CategoryID = T2.CategoryID "
            "WHERE T2.CategoryName = 'Beverages';"
        ),
        test_type="join",
        operation="lookup",
    ),
    TestCase(
        test_id="retail_world_join_supplier_highest_price",
        db_id="retail_world",
        selected_tables=["Products", "Suppliers"],
        question="What is the name of the company that supplies the product with the highest unit price?",
        expected_sql=(
            "SELECT T2.CompanyName "
            "FROM Products AS T1 "
            "INNER JOIN Suppliers AS T2 ON T1.SupplierID = T2.SupplierID "
            "ORDER BY T1.UnitPrice DESC "
            "LIMIT 1;"
        ),
        test_type="join",
        operation="max",
    ),

    # ========================================================
    # superstore
    # ========================================================
    TestCase(
        test_id="superstore_single_count_west_orders",
        db_id="superstore",
        selected_tables=["west_superstore"],
        question="How many orders were placed in the West superstore?",
        expected_sql='SELECT COUNT("Order ID") FROM west_superstore;',
        test_type="single",
        operation="count",
    ),
    TestCase(
        test_id="superstore_single_sum_west_sales",
        db_id="superstore",
        selected_tables=["west_superstore"],
        question="What is the total sales amount in the West superstore?",
        expected_sql='SELECT SUM(Sales) FROM west_superstore;',
        test_type="single",
        operation="sum",
    ),
    TestCase(
        test_id="superstore_single_avg_west_discount",
        db_id="superstore",
        selected_tables=["west_superstore"],
        question="What is the average discount in the West superstore?",
        expected_sql='SELECT AVG(Discount) FROM west_superstore;',
        test_type="single",
        operation="avg",
    ),
    TestCase(
        test_id="superstore_join_product_west",
        db_id="superstore",
        selected_tables=["west_superstore", "product"],
        question="Which product was ordered the most in the West superstore?",
        expected_sql=(
            'SELECT T2."Product Name" '
            'FROM west_superstore AS T1 '
            'INNER JOIN product AS T2 ON T1."Product ID" = T2."Product ID" '
            'GROUP BY T2."Product Name" '
            'ORDER BY SUM(T1.Quantity) DESC '
            'LIMIT 1;'
        ),
        test_type="join",
        operation="sum",
    ),

    # ========================================================
    # retails
    # ========================================================
    TestCase(
        test_id="retails_single_count_furniture",
        db_id="retails",
        selected_tables=["customer"],
        question="How many customers are in the furniture segment?",
        expected_sql="SELECT COUNT(c_custkey) FROM customer WHERE c_mktsegment = 'FURNITURE';",
        test_type="single",
        operation="count",
    ),
    TestCase(
        test_id="retails_single_avg_acctbal",
        db_id="retails",
        selected_tables=["customer"],
        question="What is the average account balance of customers?",
        expected_sql="SELECT AVG(c_acctbal) FROM customer;",
        test_type="single",
        operation="avg",
    ),
    TestCase(
        test_id="retails_single_max_order",
        db_id="retails",
        selected_tables=["orders"],
        question="Which order has the highest total price?",
        expected_sql="SELECT o_orderkey FROM orders ORDER BY o_totalprice DESC LIMIT 1;",
        test_type="single",
        operation="max",
    ),
    TestCase(
        test_id="retails_join_total_customer_orders",
        db_id="retails",
        selected_tables=["customer", "orders"],
        question="Calculate the total price of orders by Customer#000000013.",
        expected_sql=(
            "SELECT SUM(T2.o_totalprice) "
            "FROM customer AS T1 "
            "INNER JOIN orders AS T2 ON T1.c_custkey = T2.o_custkey "
            "WHERE T1.c_name = 'Customer#000000013';"
        ),
        test_type="join",
        operation="sum",
    ),
]


# ============================================================
# Utility functions
# ============================================================

def project_root() -> Path:
    return Path(__file__).resolve().parent


def clean_runtime_cache(root: Path) -> None:
    """
    Remove runtime cache that can leak old SQL/candidates.
    Does not delete DuckDB databases or source files.
    """
    targets = [
        root / ".streamlit" / "cache",
        root / "__pycache__",
        root / ".pytest_cache",
    ]

    # Remove only dangerous generated prediction caches, not necessarily HF model cache.
    dangerous_cache = root / ".cache" / "lightgbm_reranker"
    if dangerous_cache.exists():
        for child in dangerous_cache.glob("*prediction*"):
            if child.is_file():
                child.unlink()
        for child in dangerous_cache.glob("*.csv"):
            if child.is_file():
                child.unlink()

    for t in targets:
        if t.exists():
            shutil.rmtree(t, ignore_errors=True)


def find_db_path(root: Path, db_id: str) -> Optional[Path]:
    candidates = [
        root / "duckdb_databases" / f"{db_id}.duckdb",
        root / "data" / "duckdb_databases" / f"{db_id}.duckdb",
        root / f"{db_id}.duckdb",
    ]
    for c in candidates:
        if c.exists():
            return c

    matches = list(root.rglob(f"{db_id}.duckdb"))
    return matches[0] if matches else None


def quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def get_schema(db_path: Path) -> Dict[str, List[str]]:
    con = duckdb.connect(str(db_path))
    try:
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        schema = {}
        for t in tables:
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({quote_ident(t)})").fetchall()]
            schema[t] = cols
        return schema
    finally:
        con.close()


def selected_schema_exists(schema: Dict[str, List[str]], selected_tables: List[str]) -> Tuple[bool, str]:
    missing = [t for t in selected_tables if t not in schema]
    if missing:
        return False, f"Missing selected tables: {missing}"
    return True, ""


def execute_df(db_path: Path, sql: str, limit: Optional[int] = None) -> pd.DataFrame:
    sql = normalize_sql_for_duckdb(sql)
    if limit is not None:
        sql = f"SELECT * FROM ({sql.rstrip(';')}) AS q LIMIT {int(limit)}"
    con = duckdb.connect(str(db_path))
    try:
        return con.execute(sql).fetchdf()
    finally:
        con.close()


def explain_sql(db_path: Path, sql: str) -> Tuple[bool, str]:
    sql = normalize_sql_for_duckdb(sql)
    con = duckdb.connect(str(db_path))
    try:
        con.execute("EXPLAIN " + sql.rstrip(";")).fetchall()
        return True, ""
    except Exception as e:
        return False, str(e)
    finally:
        con.close()


def normalize_sql_for_duckdb(sql: str) -> str:
    """
    DuckDB prefers double quotes for identifiers.
    The trained model often emits backticks.
    This keeps runtime fair by converting backticks to double quotes.
    """
    if not sql:
        return sql
    return sql.replace("`", '"').strip()


def canonicalize_df(df: pd.DataFrame) -> List[List[str]]:
    """
    Compare result sets fairly:
    - convert all values to strings
    - normalize NaN/None
    - ignore column names
    - sort rows for order-insensitive comparison
    """
    if df is None:
        return []

    temp = df.copy()

    # Normalize values
    temp = temp.where(pd.notna(temp), None)

    rows = []
    for row in temp.itertuples(index=False, name=None):
        cleaned = []
        for v in row:
            if v is None:
                cleaned.append("<NULL>")
            elif isinstance(v, float):
                cleaned.append(f"{v:.10g}")
            else:
                cleaned.append(str(v).strip().lower())
        rows.append(cleaned)

    rows = sorted(rows)
    return rows


def compare_results(generated_df: pd.DataFrame, expected_df: pd.DataFrame) -> bool:
    return canonicalize_df(generated_df) == canonicalize_df(expected_df)


def result_preview(df: Optional[pd.DataFrame], max_rows: int = 3) -> str:
    if df is None:
        return ""
    try:
        return df.head(max_rows).to_json(orient="records", force_ascii=False)
    except Exception:
        return str(df.head(max_rows).to_dict(orient="records"))


def crude_invalid_token_check(sql: str) -> Tuple[bool, str]:
    """
    Hard detect common broken cases before DuckDB.
    """
    if not sql or not sql.strip():
        return False, "empty_sql"

    # prompt leakage
    leak_terms = ["Question:", "Relevant Table Schemas:", "Intent Mapping:", "SQL generation rules:"]
    if any(term.lower() in sql.lower() for term in leak_terms):
        return False, "prompt_leakage"

    # undefined aliases are mostly caught by DuckDB, but this gives a clearer reason.
    aliases_defined = set(re.findall(r"\b(?:FROM|JOIN)\s+[\w\"`\[\]. ]+\s+(?:AS\s+)?(T\d+)\b", sql, flags=re.I))
    aliases_used = set(re.findall(r"\b(T\d+)\.", sql))
    undefined = sorted(aliases_used - aliases_defined)
    if undefined:
        return False, f"undefined_aliases={undefined}"

    return True, ""


# ============================================================
# Engine integration
# ============================================================

def load_system(root: Path, no_ml: bool = False):
    """
    Loads the user's Text-to-SQL system.

    This expects the existing project to expose functions from text2sql_engine.py:
    - load_spacy_ner
    - load_codet5
    - load_value_index
    - load_lightgbm_reranker
    - run_text_to_sql
    """
    sys.path.insert(0, str(root))

    import text2sql_engine as engine  # noqa

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    print("Loading spaCy NER...")
    nlp = engine.load_spacy_ner(hf_token=hf_token)

    print("Loading CodeT5...")
    tokenizer, codet5_model, device = engine.load_codet5(hf_token=hf_token)

    print("Loading value index...")
    value_index = engine.load_value_index(hf_token=hf_token)

    ml_reranker = None
    feature_cols = []

    if not no_ml:
        try:
            print("Loading LightGBM reranker...")
            ml_reranker, feature_cols = engine.load_lightgbm_reranker(hf_token=hf_token)
        except Exception as e:
            print(f"WARNING: Could not load LightGBM reranker. Falling back without ML. Reason: {e}")
            ml_reranker = None
            feature_cols = []

    return {
        "engine": engine,
        "nlp": nlp,
        "tokenizer": tokenizer,
        "codet5_model": codet5_model,
        "device": device,
        "value_index": value_index,
        "ml_reranker": ml_reranker,
        "feature_cols": feature_cols,
    }


def generate_sql(system: Dict[str, Any], case: TestCase, db_path: Path) -> Dict[str, Any]:
    """
    Calls the app pipeline.

    If ML reranker is unavailable, we still try to call run_text_to_sql. If the
    project requires ML, this will fail and the error is reported.
    """
    engine = system["engine"]

    if system["ml_reranker"] is None:
        # Some versions of run_text_to_sql require a non-null ML model.
        # If your fixed system supports no-ML mode, it should handle this.
        pass

    return engine.run_text_to_sql(
        question=case.question,
        db_id=case.db_id,
        db_path=str(db_path),
        selected_tables=case.selected_tables,
        nlp=system["nlp"],
        tokenizer=system["tokenizer"],
        codet5_model=system["codet5_model"],
        device=system["device"],
        value_index=system["value_index"],
        ml_reranker=system["ml_reranker"],
        feature_cols=system["feature_cols"],
        max_rows=1000,
    )


# ============================================================
# Regression runner
# ============================================================

def run_one_case(root: Path, system: Dict[str, Any], case: TestCase) -> Dict[str, Any]:
    row = asdict(case)

    db_path = find_db_path(root, case.db_id)
    row["db_path"] = str(db_path) if db_path else ""
    row["golden_valid"] = False
    row["golden_error"] = ""
    row["generated_sql"] = ""
    row["generated_success_flag"] = False
    row["generated_error"] = ""
    row["schema_valid"] = False
    row["schema_error"] = ""
    row["execution_success"] = False
    row["execution_error"] = ""
    row["result_match"] = False
    row["expected_preview"] = ""
    row["generated_preview"] = ""
    row["selected_schema_ok"] = False
    row["selected_schema_error"] = ""

    if db_path is None:
        row["golden_error"] = f"DB not found for db_id={case.db_id}"
        return row

    try:
        schema = get_schema(db_path)
        ok, msg = selected_schema_exists(schema, case.selected_tables)
        row["selected_schema_ok"] = ok
        row["selected_schema_error"] = msg
        if not ok:
            return row
    except Exception as e:
        row["selected_schema_error"] = str(e)
        return row

    # Validate and execute golden expected SQL.
    golden_ok, golden_err = explain_sql(db_path, case.expected_sql)
    row["golden_valid"] = golden_ok
    row["golden_error"] = golden_err

    if not golden_ok:
        return row

    try:
        expected_df = execute_df(db_path, case.expected_sql)
        row["expected_preview"] = result_preview(expected_df)
    except Exception as e:
        row["golden_valid"] = False
        row["golden_error"] = f"Golden execution failed: {e}"
        return row

    # Generate SQL
    try:
        output = generate_sql(system, case, db_path)
        row["generated_success_flag"] = bool(output.get("success"))
        row["generated_error"] = str(output.get("error") or "")
        row["generated_sql"] = str(output.get("best_sql") or output.get("executed_sql") or "")
    except Exception as e:
        row["generated_error"] = traceback.format_exc()
        return row

    generated_sql = row["generated_sql"]

    # Crude invalid checks
    crude_ok, crude_err = crude_invalid_token_check(generated_sql)
    if not crude_ok:
        row["schema_valid"] = False
        row["schema_error"] = crude_err
        return row

    # Schema / execution validation by DuckDB EXPLAIN
    schema_ok, schema_err = explain_sql(db_path, generated_sql)
    row["schema_valid"] = schema_ok
    row["schema_error"] = schema_err

    if not schema_ok:
        return row

    # Execute generated SQL
    try:
        generated_df = execute_df(db_path, generated_sql)
        row["execution_success"] = True
        row["generated_preview"] = result_preview(generated_df)
        row["result_match"] = compare_results(generated_df, expected_df)
    except Exception as e:
        row["execution_success"] = False
        row["execution_error"] = str(e)

    return row


def summarize(results: pd.DataFrame) -> Dict[str, Any]:
    valid_gold = results[results["golden_valid"] == True].copy()

    def rate(col: str, frame: pd.DataFrame) -> float:
        if len(frame) == 0:
            return 0.0
        return float(frame[col].fillna(False).mean())

    summary = {
        "total_cases": int(len(results)),
        "valid_golden_cases": int(len(valid_gold)),
        "schema_valid_rate": rate("schema_valid", valid_gold),
        "execution_success_rate": rate("execution_success", valid_gold),
        "result_accuracy": rate("result_match", valid_gold),
    }

    by_type = {}
    for t, g in valid_gold.groupby("test_type"):
        by_type[t] = {
            "cases": int(len(g)),
            "schema_valid_rate": rate("schema_valid", g),
            "execution_success_rate": rate("execution_success", g),
            "result_accuracy": rate("result_match", g),
        }

    by_operation = {}
    for op, g in valid_gold.groupby("operation"):
        by_operation[op] = {
            "cases": int(len(g)),
            "schema_valid_rate": rate("schema_valid", g),
            "execution_success_rate": rate("execution_success", g),
            "result_accuracy": rate("result_match", g),
        }

    by_db = {}
    for db, g in valid_gold.groupby("db_id"):
        by_db[db] = {
            "cases": int(len(g)),
            "schema_valid_rate": rate("schema_valid", g),
            "execution_success_rate": rate("execution_success", g),
            "result_accuracy": rate("result_match", g),
        }

    summary["by_type"] = by_type
    summary["by_operation"] = by_operation
    summary["by_db"] = by_db

    return summary


def write_markdown_report(results: pd.DataFrame, summary: Dict[str, Any], path: Path) -> None:
    lines = []
    lines.append("# Text-to-SQL Regression Test Report\n")
    lines.append("## Overall accuracy\n")
    lines.append(f"- Total cases: **{summary['total_cases']}**")
    lines.append(f"- Valid golden cases: **{summary['valid_golden_cases']}**")
    lines.append(f"- Schema valid rate: **{summary['schema_valid_rate']:.2%}**")
    lines.append(f"- Execution success rate: **{summary['execution_success_rate']:.2%}**")
    lines.append(f"- Result accuracy: **{summary['result_accuracy']:.2%}**\n")

    lines.append("## Accuracy by test type\n")
    for k, v in summary["by_type"].items():
        lines.append(
            f"- **{k}**: cases={v['cases']}, "
            f"schema={v['schema_valid_rate']:.2%}, "
            f"execution={v['execution_success_rate']:.2%}, "
            f"result={v['result_accuracy']:.2%}"
        )
    lines.append("")

    lines.append("## Accuracy by operation\n")
    for k, v in summary["by_operation"].items():
        lines.append(
            f"- **{k}**: cases={v['cases']}, "
            f"schema={v['schema_valid_rate']:.2%}, "
            f"execution={v['execution_success_rate']:.2%}, "
            f"result={v['result_accuracy']:.2%}"
        )
    lines.append("")

    lines.append("## Accuracy by database\n")
    for k, v in summary["by_db"].items():
        lines.append(
            f"- **{k}**: cases={v['cases']}, "
            f"schema={v['schema_valid_rate']:.2%}, "
            f"execution={v['execution_success_rate']:.2%}, "
            f"result={v['result_accuracy']:.2%}"
        )
    lines.append("")

    lines.append("## Failed cases\n")
    failed = results[(results["golden_valid"] == True) & (results["result_match"] == False)]
    if failed.empty:
        lines.append("No failed valid-golden cases.\n")
    else:
        for _, r in failed.iterrows():
            lines.append(f"### {r['test_id']}")
            lines.append(f"- DB: `{r['db_id']}`")
            lines.append(f"- Tables: `{r['selected_tables']}`")
            lines.append(f"- Type: `{r['test_type']}`")
            lines.append(f"- Operation: `{r['operation']}`")
            lines.append(f"- Question: {r['question']}")
            lines.append(f"- Expected SQL:\n```sql\n{r['expected_sql']}\n```")
            lines.append(f"- Generated SQL:\n```sql\n{r['generated_sql']}\n```")
            lines.append(f"- Schema valid: `{r['schema_valid']}`")
            lines.append(f"- Execution success: `{r['execution_success']}`")
            if r.get("schema_error"):
                lines.append(f"- Schema error: `{r['schema_error']}`")
            if r.get("execution_error"):
                lines.append(f"- Execution error: `{r['execution_error']}`")
            if r.get("generated_error"):
                lines.append(f"- Generated error: `{r['generated_error']}`")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="regression_test_results.csv")
    parser.add_argument("--summary-json", default="regression_test_summary.json")
    parser.add_argument("--report-md", default="regression_test_results.md")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--no-ml", action="store_true")
    parser.add_argument("--clean-runtime-cache", action="store_true")
    args = parser.parse_args()

    root = project_root()

    if args.clean_runtime_cache:
        clean_runtime_cache(root)

    cases = TEST_CASES
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    system = load_system(root, no_ml=args.no_ml)

    rows = []
    for i, case in enumerate(cases, start=1):
        print(f"\n[{i}/{len(cases)}] {case.test_id}")
        print(f"DB={case.db_id} | tables={case.selected_tables} | op={case.operation} | type={case.test_type}")
        print(f"Q: {case.question}")

        row = run_one_case(root, system, case)
        rows.append(row)

        print(f"Generated SQL: {row.get('generated_sql')}")
        print(
            f"golden={row.get('golden_valid')} | "
            f"schema={row.get('schema_valid')} | "
            f"exec={row.get('execution_success')} | "
            f"match={row.get('result_match')}"
        )

        if not row.get("result_match"):
            err = row.get("schema_error") or row.get("execution_error") or row.get("generated_error") or row.get("golden_error")
            if err:
                print(f"Reason: {str(err)[:500]}")

    results = pd.DataFrame(rows)
    results.to_csv(args.out, index=False)

    summary = summarize(results)
    Path(args.summary_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_report(results, summary, Path(args.report_md))

    print("\n==============================")
    print("REGRESSION TEST SUMMARY")
    print("==============================")
    print(f"Total cases: {summary['total_cases']}")
    print(f"Valid golden cases: {summary['valid_golden_cases']}")
    print(f"Schema valid rate: {summary['schema_valid_rate']:.2%}")
    print(f"Execution success rate: {summary['execution_success_rate']:.2%}")
    print(f"Result accuracy: {summary['result_accuracy']:.2%}")
    print("==============================")
    print(f"Wrote: {args.out}")
    print(f"Wrote: {args.summary_json}")
    print(f"Wrote: {args.report_md}")

    # Useful exit code for CI / automated agent testing.
    if summary["result_accuracy"] < 0.80:
        sys.exit(1)


if __name__ == "__main__":
    main()
