from pathlib import Path
import os

# ============================================================
# Hugging Face production resources
# ============================================================

SPACY_NER_REPO_ID = "BernardJoshua/text-to-sql-spacy-ner-final"
CODET5_REPO_ID = "BernardJoshua/codet5-small-text-to-sql-prompt-final_model"
LIGHTGBM_RERANKER_REPO_ID = "BernardJoshua/text-to-sql-lightgbm-reranker"

# Your value index is uploaded as a Hugging Face dataset.
VALUE_INDEX_REPO_ID = "BernardJoshua/spacy-ner-dataset"
VALUE_INDEX_FILENAME = "value_index.jsonl"

# ============================================================
# Runtime config
# ============================================================

SQL_DIALECT = "duckdb"

NUM_CANDIDATES = int(os.environ.get("NUM_CANDIDATES", "30"))
MAX_INPUT_LENGTH = int(os.environ.get("MAX_INPUT_LENGTH", "1024"))
MAX_OUTPUT_LENGTH = int(os.environ.get("MAX_OUTPUT_LENGTH", "512"))

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache"
DB_DIR = BASE_DIR / "duckdb_databases"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)
