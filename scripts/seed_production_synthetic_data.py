import argparse
import csv
import os
import random
import re
import shutil
from datetime import date, timedelta
from pathlib import Path

import duckdb


PROJECT_DIR = Path(__file__).resolve().parents[1]
DB_DIR = PROJECT_DIR / "duckdb_databases"
CATALOG_CSV = PROJECT_DIR / "synthetic_query_catalog.csv"
CATALOG_MD = PROJECT_DIR / "SYNTHETIC_QUERY_CATALOG.md"

TARGET_ROWS = 10_000
BATCH_SIZE = 1_000
random.seed(42)


FIRST_NAMES = [
    "Avery", "Jordan", "Taylor", "Morgan", "Riley", "Cameron", "Olivia", "Ethan",
    "Sophia", "Liam", "Maya", "Noah", "Aria", "Daniel", "Isabella", "Lucas",
]

LAST_NAMES = [
    "Carter", "Bennett", "Reyes", "Morgan", "Patel", "Williams", "Nguyen",
    "Thompson", "Brooks", "Garcia", "Wilson", "Tan", "Lim", "Rodriguez",
]

CITIES = [
    "Portland", "Austin", "Seattle", "Denver", "Chicago", "New York City",
    "Los Angeles", "Boston", "Phoenix", "San Diego", "Dallas", "Houston",
    "Miami", "Nashville", "Atlanta", "San Francisco",
]

STATES = [
    "Oregon", "Texas", "Washington", "Colorado", "Illinois", "New York",
    "California", "Massachusetts", "Arizona", "Florida", "Georgia",
]

COUNTIES = [
    "Multnomah County", "Travis County", "King County", "Denver County",
    "Cook County", "Los Angeles County", "Suffolk County",
]

REGIONS = [
    "Pacific Northwest", "Southwest", "West", "Midwest", "Northeast", "Southeast",
]

ISSUES = [
    "Billing disputes", "Account opening", "Fraud or scam", "Incorrect information",
    "Payment processing", "Late fee dispute", "Unauthorized transaction",
    "Customer service issue",
]

PRODUCTS = [
    "18th Century Vintage Horse Carriage", "Aurora Coffee Beans",
    "Northwind Chai", "Executive Office Chair", "Wireless Keyboard",
    "Premium Green Tea", "Beverage Sampler Pack", "Classic Road Bike",
    "Eco Laptop Stand", "Organic Breakfast Cereal", "Noise Cancelling Headphones",
]

CATEGORIES = [
    "Beverages", "Office Supplies", "Electronics", "Furniture", "Grocery",
    "Automotive", "Health", "Sports",
]

RESTAURANTS = [
    "Plearn-Thai Cuisine", "Golden Lantern Bistro", "Pacific Grill House",
    "Urban Noodle Kitchen", "Blue Harbor Cafe",
]

CUSTOMERS = [
    "Aimee Bixby", "Nancy Davolio", "Robert King", "Margaret Peacock",
    "Andrew Fuller", "Janet Leverling", "Michael Suyama", "Laura Callahan",
]

EMPLOYEES = [
    "Olivia Carter", "Ethan Brooks", "Sophia Nguyen", "Liam Bennett",
    "Maya Thompson", "Daniel Reyes",
]


def q(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def norm(x: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def is_int_type(t: str) -> bool:
    t = str(t).upper()
    return "INT" in t or t in {"BIGINT", "SMALLINT", "TINYINT"}


def is_float_type(t: str) -> bool:
    t = str(t).upper()
    return any(x in t for x in ["DOUBLE", "FLOAT", "REAL", "DECIMAL", "NUMERIC"])


def is_date_type(t: str) -> bool:
    t = str(t).upper()
    return "DATE" in t or "TIME" in t


def is_text_type(t: str) -> bool:
    t = str(t).upper()
    return any(x in t for x in ["CHAR", "TEXT", "VARCHAR", "STRING"])


def get_tables(con):
    return [r[0] for r in con.execute("SHOW TABLES").fetchall()]


def get_schema(con, table):
    rows = con.execute(f"PRAGMA table_info({q(table)})").fetchall()
    # DuckDB: cid, name, type, notnull, dflt_value, pk
    return [
        {
            "cid": r[0],
            "name": r[1],
            "type": r[2],
            "notnull": r[3],
            "default": r[4],
            "pk": r[5],
        }
        for r in rows
    ]


def row_count(con, table):
    return con.execute(f"SELECT COUNT(*) FROM {q(table)}").fetchone()[0]


def find_table(tables, candidates):
    table_map = {norm(t): t for t in tables}
    for cand in candidates:
        if norm(cand) in table_map:
            return table_map[norm(cand)]
    for t in tables:
        nt = norm(t)
        for cand in candidates:
            if norm(cand) in nt or nt in norm(cand):
                return t
    return None


def find_col(schema, candidates):
    col_map = {norm(c["name"]): c["name"] for c in schema}
    for cand in candidates:
        if norm(cand) in col_map:
            return col_map[norm(cand)]
    for c in schema:
        nc = norm(c["name"])
        for cand in candidates:
            ncan = norm(cand)
            if ncan and (ncan in nc or nc in ncan):
                return c["name"]
    return None


def col_type(schema, col):
    for c in schema:
        if c["name"] == col:
            return c["type"]
    return "TEXT"


def make_date(i):
    d = date(2015, 1, 1) + timedelta(days=i % 3650)
    return str(d)


def default_value(db_id, table, column, dtype, i, key_pools):
    nc = norm(column)
    nt = str(dtype).upper()

    if nc in key_pools:
        return key_pools[nc][i % len(key_pools[nc])]

    if is_int_type(nt):
        if "quantity" in nc or "qty" in nc:
            return random.randint(1, 25)
        if "age" in nc:
            return random.randint(18, 75)
        if "cylinder" in nc:
            return random.choice([4, 6, 8])
        if "year" in nc:
            return random.randint(2015, 2024)
        if "id" in nc or "key" in nc or "code" in nc:
            return 1_000_000 + i
        return random.randint(1, 10_000)

    if is_float_type(nt):
        if "price" in nc:
            return round(random.uniform(18_000, 95_000), 2)
        if "amount" in nc or "sales" in nc or "revenue" in nc:
            return round(random.uniform(50, 2_500), 2)
        if "cost" in nc:
            return round(random.uniform(10, 900), 2)
        if "discount" in nc:
            return round(random.uniform(0.0, 0.25), 2)
        return round(random.uniform(1, 10_000), 2)

    if is_date_type(nt):
        return make_date(i)

    # Text-like values
    if "city" in nc:
        return CITIES[i % len(CITIES)]
    if "state" in nc:
        return STATES[i % len(STATES)]
    if "county" in nc:
        return COUNTIES[i % len(COUNTIES)]
    if "region" in nc:
        return REGIONS[i % len(REGIONS)]
    if "issue" in nc:
        return ISSUES[i % len(ISSUES)]
    if "category" in nc:
        return CATEGORIES[i % len(CATEGORIES)]
    if "productname" in nc or nc == "product" or "itemname" in nc:
        return PRODUCTS[i % len(PRODUCTS)]
    if "restaurant" in nc or nc in {"name", "label"} and "restaurant" in norm(table):
        return RESTAURANTS[i % len(RESTAURANTS)]
    if "customer" in nc:
        return CUSTOMERS[i % len(CUSTOMERS)]
    if "employee" in nc or "salesperson" in nc:
        return EMPLOYEES[i % len(EMPLOYEES)]
    if "first" in nc:
        return FIRST_NAMES[i % len(FIRST_NAMES)]
    if "last" in nc:
        return LAST_NAMES[i % len(LAST_NAMES)]
    if "email" in nc:
        return f"user{i:06d}@example.com"
    if "phone" in nc:
        return f"555-{100 + (i % 900)}-{1000 + (i % 9000)}"
    if "date" in nc or "year" in nc or "orderdate" in nc:
        return make_date(i)
    if "address" in nc:
        return f"{100 + i % 9000} Market Street"
    if "sex" in nc or "gender" in nc:
        return random.choice(["Male", "Female", "Other"])
    if "id" in nc or "key" in nc or "code" in nc:
        prefix = re.sub(r"[^A-Z]", "", table.upper())[:4] or "KEY"
        return f"{prefix}-{i:06d}"

    return f"{table.replace(' ', '_')}_{column.replace(' ', '_')}_{i:06d}"


def build_key_pools(all_schemas, target_rows):
    key_pools = {}

    for table, schema in all_schemas.items():
        for c in schema:
            nc = norm(c["name"])
            if "id" in nc or "key" in nc or "code" in nc:
                dtype = c["type"]
                if is_int_type(dtype):
                    key_pools.setdefault(nc, [1_000_000 + i for i in range(target_rows)])
                else:
                    prefix = nc[:8].upper() or "KEY"
                    key_pools.setdefault(nc, [f"{prefix}-{i:06d}" for i in range(target_rows)])

    # Important cross-table aliases
    aliases = [
        ("clientid", "clientid"),
        ("customerid", "customerid"),
        ("productid", "productid"),
        ("productcode", "productcode"),
        ("orderid", "orderid"),
        ("ordernumber", "ordernumber"),
        ("employeeid", "employeeid"),
        ("salespersonid", "salespersonid"),
        ("restaurantid", "restaurantid"),
        ("categoryid", "categoryid"),
        ("partkey", "partkey"),
        ("orderkey", "orderkey"),
    ]

    for a, b in aliases:
        if a in key_pools and b not in key_pools:
            key_pools[b] = key_pools[a]
        if b in key_pools and a not in key_pools:
            key_pools[a] = key_pools[b]

    return key_pools


def insert_rows(con, table, schema, rows, db_id, key_pools):
    cols = [c["name"] for c in schema]
    placeholders = ", ".join(["?"] * len(cols))
    col_sql = ", ".join(q(c) for c in cols)
    sql = f"INSERT INTO {q(table)} ({col_sql}) VALUES ({placeholders})"

    values = []
    base = row_count(con, table)

    for offset, overrides in enumerate(rows):
        i = base + offset + 1
        row_values = []
        for c in schema:
            col = c["name"]
            if col in overrides:
                row_values.append(overrides[col])
            else:
                row_values.append(default_value(db_id, table, col, c["type"], i, key_pools))
        values.append(row_values)

    if values:
        con.executemany(sql, values)


def fill_table_to_target(con, db_id, table, schema, key_pools, target_rows):
    existing = row_count(con, table)
    needed = max(0, target_rows - existing)

    print(f"  {table}: existing={existing}, need_insert={needed}")

    inserted = 0
    while inserted < needed:
        n = min(BATCH_SIZE, needed - inserted)
        rows = [{} for _ in range(n)]
        insert_rows(con, table, schema, rows, db_id, key_pools)
        inserted += n


def typed_anchor_value(schema, col, text_value, numeric_value):
    dtype = col_type(schema, col)
    if is_int_type(dtype):
        return numeric_value
    return text_value


def seed_retail_complaints(con, tables, schemas, key_pools):
    client = find_table(tables, ["client"])
    events = find_table(tables, ["events"])
    if not client or not events:
        return None

    client_schema = schemas[client]
    events_schema = schemas[events]

    c_id = find_col(client_schema, ["client_id", "client id", "Client_ID"])
    c_city = find_col(client_schema, ["city"])
    c_first = find_col(client_schema, ["first", "first_name"])
    c_last = find_col(client_schema, ["last", "last_name"])

    e_client = find_col(events_schema, ["Client_ID", "client_id", "client id"])
    e_complaint = find_col(events_schema, ["Complaint ID", "complaint_id"])
    e_issue = find_col(events_schema, ["Issue"])
    e_product = find_col(events_schema, ["Product"])
    e_received = find_col(events_schema, ["Date received", "date_received"])
    e_sent = find_col(events_schema, ["Date sent to company", "date_sent"])

    if not all([c_id, c_city, e_client, e_complaint, e_issue]):
        return None

    client_rows = []
    event_rows = []

    for i in range(300):
        key = typed_anchor_value(client_schema, c_id, f"RC-PORT-{i:06d}", 8_800_000 + i)
        client_row = {
            c_id: key,
            c_city: "Portland",
        }
        if c_first:
            client_row[c_first] = FIRST_NAMES[i % len(FIRST_NAMES)]
        if c_last:
            client_row[c_last] = LAST_NAMES[i % len(LAST_NAMES)]

        event_row = {
            e_client: key,
            e_complaint: f"CMP-PORT-BILL-{i:06d}",
            e_issue: "Billing disputes",
        }
        if e_product:
            event_row[e_product] = "Credit card"
        if e_received:
            event_row[e_received] = "2016-06-15"
        if e_sent:
            event_row[e_sent] = "2016-06-20"

        client_rows.append(client_row)
        event_rows.append(event_row)

    insert_rows(con, client, client_schema, client_rows, "retail_complains", key_pools)
    insert_rows(con, events, events_schema, event_rows, "retail_complains", key_pools)

    return {
        "db": "retail_complains",
        "question": "Among clients from Portland, how many complaints were about Billing disputes?",
        "check_sql": f"""
SELECT COUNT(*) AS complaint_count
FROM {q(client)} AS T1
JOIN {q(events)} AS T2
  ON T1.{q(c_id)} = T2.{q(e_client)}
WHERE T1.{q(c_city)} = 'Portland'
  AND T2.{q(e_issue)} = 'Billing disputes'
""",
    }


def seed_car_retails(con, tables, schemas, key_pools):
    products = find_table(tables, ["products"])
    orderdetails = find_table(tables, ["orderdetails", "order_details"])
    if not products or not orderdetails:
        return None

    ps = schemas[products]
    os = schemas[orderdetails]

    p_key = find_col(ps, ["ProductID", "ProductCode", "product_id", "productCode"])
    o_key = find_col(os, ["ProductID", "ProductCode", "product_id", "productCode"])
    p_name = find_col(ps, ["ProductName", "Product Name", "product_name"])
    qty = find_col(os, ["QuantityOrdered", "quantityOrdered", "quantity", "qty"])

    if not all([p_key, o_key, p_name, qty]):
        return None

    key = typed_anchor_value(ps, p_key, "CR-HORSE-0001", 7_700_001)

    insert_rows(con, products, ps, [{p_key: key, p_name: "18th Century Vintage Horse Carriage"}], "car_retails", key_pools)

    rows = []
    for i in range(250):
        rows.append({
            o_key: key,
            qty: 5 + (i % 20),
        })
    insert_rows(con, orderdetails, os, rows, "car_retails", key_pools)

    return {
        "db": "car_retails",
        "question": "Calculate the total quantity ordered for 18th Century Vintage Horse Carriage.",
        "check_sql": f"""
SELECT SUM(T2.{q(qty)}) AS total_quantity
FROM {q(products)} AS T1
JOIN {q(orderdetails)} AS T2
  ON T1.{q(p_key)} = T2.{q(o_key)}
WHERE T1.{q(p_name)} = '18th Century Vintage Horse Carriage'
""",
    }


def seed_cars(con, tables, schemas, key_pools):
    data = find_table(tables, ["data"])
    price = find_table(tables, ["price"])
    if not data or not price:
        return None

    ds = schemas[data]
    ps = schemas[price]

    d_key = find_col(ds, ["Car_ID", "car_id", "id"])
    p_key = find_col(ps, ["Car_ID", "car_id", "id"])
    cylinders = find_col(ds, ["Cylinders", "cylinders"])
    p_price = find_col(ps, ["Price", "price"])
    car_name = find_col(ds, ["car_name", "Car Name", "name"])

    if not all([d_key, p_key, cylinders, p_price]):
        return None

    key = typed_anchor_value(ds, d_key, "CAR-CHEAP-0001", 6_600_001)

    drow = {d_key: key, cylinders: 4}
    if car_name:
        drow[car_name] = "Apex Eco Compact"

    prow = {p_key: key, p_price: 1000.00}

    insert_rows(con, data, ds, [drow], "cars", key_pools)
    insert_rows(con, price, ps, [prow], "cars", key_pools)

    return {
        "db": "cars",
        "question": "How many cylinders does the cheapest car have?",
        "check_sql": f"""
SELECT T1.{q(cylinders)} AS cylinders
FROM {q(data)} AS T1
JOIN {q(price)} AS T2
  ON T1.{q(d_key)} = T2.{q(p_key)}
ORDER BY T2.{q(p_price)} ASC
LIMIT 1
""",
    }


def seed_sales(con, tables, schemas, key_pools):
    sales = find_table(tables, ["Sales"])
    employees = find_table(tables, ["Employees"])
    if not sales or not employees:
        return None

    ss = schemas[sales]
    es = schemas[employees]

    e_key = find_col(es, ["EmployeeID", "employee_id", "SalesPersonID"])
    s_key = find_col(ss, ["EmployeeID", "employee_id", "SalesPersonID", "salesperson_id"])
    name = find_col(es, ["Name", "EmployeeName", "FirstName", "LastName"])
    qty = find_col(ss, ["Quantity", "Qty", "Items", "Units"])

    if not all([e_key, s_key, name, qty]):
        return None

    key = typed_anchor_value(es, e_key, "EMP-TOP-0001", 5_500_001)

    insert_rows(con, employees, es, [{e_key: key, name: "Olivia Carter"}], "sales", key_pools)

    rows = []
    for i in range(300):
        rows.append({
            s_key: key,
            qty: 500 + (i % 50),
        })
    insert_rows(con, sales, ss, rows, "sales", key_pools)

    return {
        "db": "sales",
        "question": "Which salesperson sold the most items in total?",
        "check_sql": f"""
SELECT T1.{q(name)} AS salesperson, SUM(T2.{q(qty)}) AS total_items
FROM {q(employees)} AS T1
JOIN {q(sales)} AS T2
  ON T1.{q(e_key)} = T2.{q(s_key)}
GROUP BY T1.{q(name)}
ORDER BY total_items DESC
LIMIT 1
""",
    }


def seed_restaurant(con, tables, schemas, key_pools):
    general = find_table(tables, ["generalinfo", "general_info"])
    geo = find_table(tables, ["geographic", "geography"])
    if not general or not geo:
        return None

    gs = schemas[general]
    geos = schemas[geo]

    g_key = find_col(gs, ["Restaurant_ID", "restaurant_id", "id"])
    geo_key = find_col(geos, ["Restaurant_ID", "restaurant_id", "id"])
    name = find_col(gs, ["Name", "label", "restaurant_name"])
    county = find_col(geos, ["County", "county"])
    region = find_col(geos, ["Region", "region"])

    if not all([g_key, geo_key, name, county, region]):
        return None

    key = typed_anchor_value(gs, g_key, "REST-THAI-0001", 4_400_001)

    insert_rows(con, general, gs, [{g_key: key, name: "Plearn-Thai Cuisine"}], "restaurant", key_pools)
    insert_rows(con, geo, geos, [{geo_key: key, county: "King County", region: "Pacific Northwest"}], "restaurant", key_pools)

    return {
        "db": "restaurant",
        "question": "What is the county and region of Plearn-Thai Cuisine restaurant?",
        "check_sql": f"""
SELECT T2.{q(county)} AS county, T2.{q(region)} AS region
FROM {q(general)} AS T1
JOIN {q(geo)} AS T2
  ON T1.{q(g_key)} = T2.{q(geo_key)}
WHERE T1.{q(name)} = 'Plearn-Thai Cuisine'
""",
    }


def seed_regional_sales(con, tables, schemas, key_pools):
    sales_orders = find_table(tables, ["Sales Orders", "sales_orders"])
    products = find_table(tables, ["Products"])
    if not sales_orders or not products:
        return None

    ss = schemas[sales_orders]
    ps = schemas[products]

    s_key = find_col(ss, ["ProductID", "_ProductID", "Product ID", "product_id"])
    p_key = find_col(ps, ["ProductID", "_ProductID", "Product ID", "product_id"])
    p_name = find_col(ps, ["ProductName", "Product Name", "product_name"])
    amount = find_col(ss, ["Sales Amount", "Amount", "Sales", "Revenue"])

    if not all([s_key, p_key, p_name, amount]):
        return None

    key = typed_anchor_value(ps, p_key, "PROD-AURORA-0001", 3_300_001)

    insert_rows(con, products, ps, [{p_key: key, p_name: "Aurora Coffee Beans"}], "regional_sales", key_pools)

    rows = []
    for i in range(300):
        rows.append({
            s_key: key,
            amount: 50_000 + (i * 10),
        })
    insert_rows(con, sales_orders, ss, rows, "regional_sales", key_pools)

    return {
        "db": "regional_sales",
        "question": "Which product generated the highest sales amount?",
        "check_sql": f"""
SELECT T2.{q(p_name)} AS product_name, SUM(T1.{q(amount)}) AS sales_amount
FROM {q(sales_orders)} AS T1
JOIN {q(products)} AS T2
  ON T1.{q(s_key)} = T2.{q(p_key)}
GROUP BY T2.{q(p_name)}
ORDER BY sales_amount DESC
LIMIT 1
""",
    }


def seed_retail_world(con, tables, schemas, key_pools):
    products = find_table(tables, ["Products"])
    categories = find_table(tables, ["Categories"])
    if not products or not categories:
        return None

    ps = schemas[products]
    cs = schemas[categories]

    p_cat = find_col(ps, ["CategoryID", "_CategoryID", "category_id"])
    c_cat = find_col(cs, ["CategoryID", "_CategoryID", "category_id"])
    c_name = find_col(cs, ["CategoryName", "Category Name", "category_name"])
    p_name = find_col(ps, ["ProductName", "Product Name", "product_name"])

    if not all([p_cat, c_cat, c_name, p_name]):
        return None

    key = typed_anchor_value(cs, c_cat, "CAT-BEV-0001", 2_200_001)

    insert_rows(con, categories, cs, [{c_cat: key, c_name: "Beverages"}], "retail_world", key_pools)

    rows = [
        {p_cat: key, p_name: "Premium Green Tea"},
        {p_cat: key, p_name: "Beverage Sampler Pack"},
        {p_cat: key, p_name: "Northwind Chai"},
    ]
    insert_rows(con, products, ps, rows, "retail_world", key_pools)

    return {
        "db": "retail_world",
        "question": "What are the products that belong to the beverage category?",
        "check_sql": f"""
SELECT T1.{q(p_name)} AS product_name
FROM {q(products)} AS T1
JOIN {q(categories)} AS T2
  ON T1.{q(p_cat)} = T2.{q(c_cat)}
WHERE T2.{q(c_name)} = 'Beverages'
LIMIT 20
""",
    }


def seed_superstore(con, tables, schemas, key_pools):
    west = find_table(tables, ["west_superstore"])
    product = find_table(tables, ["product"])
    if not west or not product:
        return None

    ws = schemas[west]
    ps = schemas[product]

    w_prod = find_col(ws, ["ProductID", "Product ID", "product_id"])
    p_prod = find_col(ps, ["ProductID", "Product ID", "product_id"])
    p_name = find_col(ps, ["ProductName", "Product Name", "product_name"])
    customer = find_col(ws, ["CustomerName", "Customer Name", "customer_name"])
    order_date = find_col(ws, ["OrderDate", "Order Date", "order_date"])

    if not all([w_prod, p_prod, p_name, customer, order_date]):
        return None

    key = typed_anchor_value(ps, p_prod, "SS-PROD-0001", 1_100_001)

    insert_rows(con, product, ps, [{p_prod: key, p_name: "Executive Office Chair"}], "superstore", key_pools)

    rows = []
    for i in range(120):
        rows.append({
            w_prod: key,
            customer: "Aimee Bixby",
            order_date: f"2016-{1 + (i % 12):02d}-{1 + (i % 27):02d}",
        })
    insert_rows(con, west, ws, rows, "superstore", key_pools)

    return {
        "db": "superstore",
        "question": "Please list the products ordered by Aimee Bixby in 2016.",
        "check_sql": f"""
SELECT DISTINCT T2.{q(p_name)} AS product_name
FROM {q(west)} AS T1
JOIN {q(product)} AS T2
  ON T1.{q(w_prod)} = T2.{q(p_prod)}
WHERE T1.{q(customer)} = 'Aimee Bixby'
  AND CAST(T1.{q(order_date)} AS VARCHAR) LIKE '2016%'
""",
    }


def seed_retails(con, tables, schemas, key_pools):
    # Supports either expected retail tables or TPCH-like retails schema.
    order_items = find_table(tables, ["order_items", "orderitems"])
    orders = find_table(tables, ["orders"])
    if order_items and orders:
        ois = schemas[order_items]
        os = schemas[orders]

        oi_order = find_col(ois, ["OrderID", "order_id"])
        o_order = find_col(os, ["OrderID", "order_id"])
        product = find_col(ois, ["ProductID", "product_id", "ProductName"])
        qty = find_col(ois, ["Quantity", "Qty"])
        unit_price = find_col(ois, ["UnitPrice", "unit_price", "Price"])
        order_date = find_col(os, ["OrderDate", "order_date"])

        if all([oi_order, o_order, product, qty, unit_price, order_date]):
            order_rows = []
            item_rows = []
            for i in range(500):
                order_key = typed_anchor_value(os, o_order, f"RET-ORD-2020-{i:06d}", 9_900_000 + i)
                order_rows.append({o_order: order_key, order_date: f"2020-{1 + (i % 12):02d}-{1 + (i % 27):02d}"})
                item_rows.append({
                    oi_order: order_key,
                    product: "Premium Green Tea" if not is_int_type(col_type(ois, product)) else 12345,
                    qty: 10 + (i % 20),
                    unit_price: 19.99,
                })

            insert_rows(con, orders, os, order_rows, "retails", key_pools)
            insert_rows(con, order_items, ois, item_rows, "retails", key_pools)

            return {
                "db": "retails",
                "question": "List the total sales amount for each product in 2020.",
                "check_sql": f"""
SELECT T2.{q(product)} AS product_id, SUM(T2.{q(qty)} * T2.{q(unit_price)}) AS total_sales
FROM {q(orders)} AS T1
JOIN {q(order_items)} AS T2
  ON T1.{q(o_order)} = T2.{q(oi_order)}
WHERE CAST(T1.{q(order_date)} AS VARCHAR) LIKE '2020%'
GROUP BY T2.{q(product)}
ORDER BY total_sales DESC
LIMIT 20
""",
            }

    # TPCH-style fallback
    lineitem = find_table(tables, ["lineitem"])
    part = find_table(tables, ["part"])
    orders = find_table(tables, ["orders"])
    if not all([lineitem, part, orders]):
        return None

    ls = schemas[lineitem]
    ps = schemas[part]
    os = schemas[orders]

    l_part = find_col(ls, ["l_partkey", "partkey", "part_key"])
    p_part = find_col(ps, ["p_partkey", "partkey", "part_key"])
    l_order = find_col(ls, ["l_orderkey", "orderkey", "order_key"])
    o_order = find_col(os, ["o_orderkey", "orderkey", "order_key"])
    p_name = find_col(ps, ["p_name", "name", "part_name"])
    qty = find_col(ls, ["l_quantity", "quantity"])
    price = find_col(ls, ["l_extendedprice", "extendedprice", "price"])
    order_date = find_col(os, ["o_orderdate", "orderdate", "order_date"])

    if not all([l_part, p_part, l_order, o_order, p_name, qty, price, order_date]):
        return None

    part_key = typed_anchor_value(ps, p_part, "PART-GREEN-0001", 8_100_001)

    insert_rows(con, part, ps, [{p_part: part_key, p_name: "Premium Green Tea"}], "retails", key_pools)

    order_rows = []
    line_rows = []
    for i in range(500):
        order_key = typed_anchor_value(os, o_order, f"RET-ORD-2020-{i:06d}", 8_200_000 + i)
        order_rows.append({o_order: order_key, order_date: f"2020-{1 + (i % 12):02d}-{1 + (i % 27):02d}"})
        line_rows.append({
            l_order: order_key,
            l_part: part_key,
            qty: 10 + (i % 20),
            price: 19.99 * (10 + (i % 20)),
        })

    insert_rows(con, orders, os, order_rows, "retails", key_pools)
    insert_rows(con, lineitem, ls, line_rows, "retails", key_pools)

    return {
        "db": "retails",
        "question": "List the total sales amount for each product in 2020.",
        "check_sql": f"""
SELECT T3.{q(p_name)} AS product_name, SUM(T1.{q(price)}) AS total_sales
FROM {q(lineitem)} AS T1
JOIN {q(orders)} AS T2
  ON T1.{q(l_order)} = T2.{q(o_order)}
JOIN {q(part)} AS T3
  ON T1.{q(l_part)} = T3.{q(p_part)}
WHERE CAST(T2.{q(order_date)} AS VARCHAR) LIKE '2020%'
GROUP BY T3.{q(p_name)}
ORDER BY total_sales DESC
LIMIT 20
""",
    }


CUSTOM_SEEDERS = {
    "retail_complains": seed_retail_complaints,
    "car_retails": seed_car_retails,
    "cars": seed_cars,
    "sales": seed_sales,
    "restaurant": seed_restaurant,
    "regional_sales": seed_regional_sales,
    "retail_world": seed_retail_world,
    "superstore": seed_superstore,
    "retails": seed_retails,
}


def write_catalog(catalog_rows):
    with open(CATALOG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["db", "question", "check_sql", "result_preview"],
        )
        writer.writeheader()
        for row in catalog_rows:
            writer.writerow(row)

    with open(CATALOG_MD, "w", encoding="utf-8") as f:
        f.write("# Synthetic Query Catalog\n\n")
        f.write("Use these questions after running the seeding script.\n\n")
        for row in catalog_rows:
            f.write(f"## {row['db']}\n\n")
            f.write(f"**Question:** {row['question']}\n\n")
            f.write("```sql\n")
            f.write(row["check_sql"].strip())
            f.write("\n```\n\n")
            f.write("**Result preview:**\n\n")
            f.write("```text\n")
            f.write(str(row.get("result_preview", "")).strip())
            f.write("\n```\n\n")


def seed_database(db_path, target_rows, backup):
    db_id = db_path.stem
    print(f"\n=== Seeding {db_id} ===")

    if backup:
        backup_dir = DB_DIR / "_backups"
        backup_dir.mkdir(exist_ok=True)
        backup_path = backup_dir / f"{db_path.stem}.before_synthetic_seed.duckdb"
        if not backup_path.exists():
            shutil.copy2(db_path, backup_path)
            print(f"  Backup created: {backup_path}")

    con = duckdb.connect(str(db_path), read_only=False)

    try:
        tables = get_tables(con)
        schemas = {t: get_schema(con, t) for t in tables}
        key_pools = build_key_pools(schemas, target_rows)

        con.execute("BEGIN TRANSACTION")

        catalog_row = None
        seeder = CUSTOM_SEEDERS.get(db_id)
        if seeder:
            catalog_row = seeder(con, tables, schemas, key_pools)

        for table in tables:
            fill_table_to_target(con, db_id, table, schemas[table], key_pools, target_rows)

        con.execute("COMMIT")

        # Verify seeded query after commit.
        if catalog_row and catalog_row.get("check_sql"):
            try:
                preview = con.execute(catalog_row["check_sql"]).fetchall()[:10]
                catalog_row["result_preview"] = str(preview)
            except Exception as e:
                catalog_row["result_preview"] = f"Verification failed: {e}"

        return catalog_row

    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-dir", default=str(DB_DIR))
    parser.add_argument("--target-rows", type=int, default=TARGET_ROWS)
    parser.add_argument("--backup", action="store_true")
    args = parser.parse_args()

    db_dir = Path(args.db_dir)
    db_files = sorted(db_dir.glob("*.duckdb"))

    if not db_files:
        raise FileNotFoundError(f"No .duckdb files found in {db_dir}")

    catalog_rows = []

    for db_path in db_files:
        row = seed_database(db_path, args.target_rows, args.backup)
        if row:
            catalog_rows.append(row)

    write_catalog(catalog_rows)

    print("\nDone.")
    print(f"Wrote: {CATALOG_CSV}")
    print(f"Wrote: {CATALOG_MD}")
    print("\nNow run:")
    print("python3 run_text2sql_regression_tests.py --out-dir .")


if __name__ == "__main__":
    main()
