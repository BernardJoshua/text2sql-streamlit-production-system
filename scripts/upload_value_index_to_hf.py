from pathlib import Path
from huggingface_hub import HfApi, create_repo, login


login()

file_path = Path("data/_extracted_duckdb_bundle/duckdb_text2sql_bundle/value_index/value_index.jsonl")

if not file_path.exists():
    raise FileNotFoundError(
        f"Value index not found: {file_path}\n"
        "Run: python scripts/build_duckdb_from_bundle.py first, or change file_path."
    )

repo_id = "BernardJoshua/spacy-ner-dataset"

create_repo(
    repo_id=repo_id,
    repo_type="dataset",
    private=True,
    exist_ok=True,
)

api = HfApi()

api.upload_file(
    path_or_fileobj=str(file_path),
    path_in_repo="value_index.jsonl",
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Upload value index dataset",
)

print(f"Uploaded successfully to: https://huggingface.co/datasets/{repo_id}")
