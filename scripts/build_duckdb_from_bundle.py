from pathlib import Path
import zipfile
import shutil
import re
import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DB_DIR = ROOT / "duckdb_databases"
EXTRACT_DIR = DATA_DIR / "_extracted_duckdb_bundle"

BUNDLE_ZIP = DATA_DIR / "duckdb_text2sql_bundle.zip"


def normalize_identifier(x):
    return str(x).strip().strip('"').strip("`").strip("[").strip("]").lower()


def patch_duplicate_columns_for_duckdb(sql_text):
    """
    DuckDB treats identifiers case-insensitively.
    Some generated seed scripts contain duplicate columns like:
        county
        County

    This function keeps the first column name and renames later duplicates:
        County -> County__dup2

    It patches both CREATE TABLE column definitions and matching INSERT column lists.
    """

    create_pattern = re.compile(
        r'CREATE\s+TABLE\s+"(?P<table>[^"]+)"\s*\((?P<body>.*?)\);',
        re.IGNORECASE | re.DOTALL,
    )

    replacements = []
    table_column_maps = {}

    for match in create_pattern.finditer(sql_text):
        table = match.group("table")
        body = match.group("body")

        lines = body.splitlines()
        seen = {}
        rename_map = []
        new_lines = []
        changed = False

        for line in lines:
            col_match = re.match(
                r'(?P<prefix>\s*)"(?P<col>[^"]+)"(?P<rest>\s+[^,]+,?\s*)$',
                line,
            )

            if not col_match:
                new_lines.append(line)
                continue

            col = col_match.group("col")
            norm = normalize_identifier(col)

            seen[norm] = seen.get(norm, 0) + 1

            new_col = col

            if seen[norm] > 1:
                new_col = f"{col}__dup{seen[norm]}"
                changed = True

            rename_map.append((col, new_col))

            new_lines.append(
                f'{col_match.group("prefix")}"{new_col}"{col_match.group("rest")}'
            )

        if changed:
            new_body = "\n".join(new_lines)
            new_create = f'CREATE TABLE "{table}" ({new_body}\n);'
            replacements.append((match.start(), match.end(), new_create))
            table_column_maps[table] = rename_map

    for start, end, new_create in reversed(replacements):
        sql_text = sql_text[:start] + new_create + sql_text[end:]

    for table, rename_map in table_column_maps.items():
        renamed_cols = [new for old, new in rename_map]
        quoted_cols = ", ".join(f'"{c}"' for c in renamed_cols)

        insert_pattern = re.compile(
            r'(INSERT\s+INTO\s+"' + re.escape(table) + r'"\s*)'
            r'\((?P<cols>.*?)\)'
            r'(\s+VALUES)',
            re.IGNORECASE | re.DOTALL,
        )

        sql_text = insert_pattern.sub(
            lambda m: m.group(1) + "(" + quoted_cols + ")" + m.group(3),
            sql_text,
        )

    return sql_text


if not BUNDLE_ZIP.exists():
    raise FileNotFoundError(
        f"Missing bundle zip: {BUNDLE_ZIP}\n"
        "Put duckdb_text2sql_bundle.zip inside the data/ folder."
    )

if EXTRACT_DIR.exists():
    shutil.rmtree(EXTRACT_DIR)

EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

print("Extracting:", BUNDLE_ZIP)

with zipfile.ZipFile(BUNDLE_ZIP, "r") as z:
    z.extractall(EXTRACT_DIR)

sql_root_candidates = list(EXTRACT_DIR.rglob("sql"))

if not sql_root_candidates:
    raise FileNotFoundError("Could not find sql/ folder inside bundle zip.")

SQL_DIR = sql_root_candidates[0]

sql_files = sorted(SQL_DIR.glob("*_duckdb_seeded_corrected.sql"))

if not sql_files:
    raise FileNotFoundError(f"No *_duckdb_seeded_corrected.sql files found in {SQL_DIR}")

print("SQL files found:", len(sql_files))

for sql_file in sql_files:
    db_id = sql_file.name.replace("_duckdb_seeded_corrected.sql", "")
    db_path = DB_DIR / f"{db_id}.duckdb"

    if db_path.exists():
        db_path.unlink()

    print(f"Building {db_path.name} from {sql_file.name}")

    sql_text = sql_file.read_text(encoding="utf-8")
    sql_text = patch_duplicate_columns_for_duckdb(sql_text)

    conn = duckdb.connect(str(db_path))
    try:
        conn.execute(sql_text)
    finally:
        conn.close()

print("Done.")
print("DuckDB files written to:", DB_DIR)
