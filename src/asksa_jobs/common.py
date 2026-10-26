"""Shared infrastructure for Ask SA jobs: config, secrets, Redshift, S3."""

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Optional, Sequence

import boto3
import pandas as pd
import redshift_connector
import yaml

logger = logging.getLogger(__name__)

CONFIG_FILE_NAME = "variables.yaml"
QUERIES_DIR_NAME = "queries"
PROJECT_ROOT_ENV = "ASKSA_PROJECT_ROOT"


@dataclass(frozen=True)
class Settings:
    """Project settings loaded from variables.yaml."""

    root: Path
    region_name: str
    bucket_name: str
    s3_project_dir: str
    update_table_dir: str
    redshift_secret_name: str
    host: str
    port: int
    database: str
    copy_role_arn: str
    hf_secret_name: str
    hf_secret_key: str
    sentiment_table: str
    ner_table: str


def find_project_root(start: Optional[Path] = None) -> Path:
    """Return the folder that contains variables.yaml.

    ASKSA_PROJECT_ROOT env var wins if set. Otherwise search upwards from
    `start` (default: cwd), so it works both from the repo root (container)
    and from notebooks/ (SageMaker Studio).
    """
    env_root = os.getenv(PROJECT_ROOT_ENV)
    if env_root:
        return Path(env_root).resolve()

    path = Path(start or Path.cwd()).resolve()
    for candidate in [path, *path.parents]:
        if (candidate / CONFIG_FILE_NAME).exists():
            return candidate
    raise FileNotFoundError(
        f"{CONFIG_FILE_NAME} not found in {path} or any parent folder"
    )


def load_settings(root: Optional[Path] = None) -> Settings:
    """Load variables.yaml into a Settings object."""
    root = Path(root).resolve() if root else find_project_root()
    with open(root / CONFIG_FILE_NAME, "r") as f:
        cfg = yaml.safe_load(f)

    settings = Settings(
        root=root,
        region_name=cfg["REGION_NAME"],
        bucket_name=cfg["BUCKET_NAME"],
        s3_project_dir=cfg["S3_PROJECT_DIR"],
        update_table_dir=cfg["UPDATE_TABLE_DIR"],
        redshift_secret_name=cfg["REDSHIFT_SECRET_NAME"],
        host=cfg["HOST"],
        port=int(cfg["PORT"]),
        database=cfg["DATABASE"],
        copy_role_arn=cfg["COPY_ROLE_ARN"],
        hf_secret_name=cfg["HF_SECRET_NAME"],
        hf_secret_key=cfg["HF_SECRET_KEY"],
        sentiment_table=cfg["SENTIMENT_TABLE"],
        ner_table=cfg["NER_TABLE"],
    )
    logger.info(f"Settings loaded from {root / CONFIG_FILE_NAME}")
    return settings


@lru_cache(maxsize=None)
def get_secret(secret_name: str, region_name: str) -> dict:
    """Read a JSON secret from Secrets Manager (cached per process)."""
    client = boto3.session.Session().client(
        service_name="secretsmanager", region_name=region_name
    )
    response = client.get_secret_value(SecretId=secret_name)
    secret = response.get("SecretString") or response["SecretBinary"]
    return json.loads(secret)


def export_hf_token(settings: Settings) -> str:
    """Put the Hugging Face token into HF_TOKEN and return it.

    transformers / huggingface_hub read HF_TOKEN from the environment,
    which is needed for gated models such as google/gemma-2-9b-it.
    """
    secret = get_secret(settings.hf_secret_name, settings.region_name)
    token = secret[settings.hf_secret_key]
    os.environ["HF_TOKEN"] = token
    return token


def redshift_connect(settings: Settings) -> redshift_connector.Connection:
    """Open a new Redshift connection using credentials from the secret."""
    creds = get_secret(settings.redshift_secret_name, settings.region_name)
    return redshift_connector.connect(
        host=settings.host,
        port=settings.port,
        database=settings.database,
        user=creds["username"],
        password=creds["password"],
    )


def read_query(settings: Settings, file_name: str) -> str:
    """Read a SQL file from the queries/ folder."""
    return (settings.root / QUERIES_DIR_NAME / file_name).read_text()


def _column_name(name) -> str:
    return name.decode() if isinstance(name, bytes) else str(name)


def fetch_df(settings: Settings, sql: str) -> pd.DataFrame:
    """Run a query and return the result as a DataFrame.

    An empty result is always an empty DataFrame with the query's columns
    (older redshift_connector versions return None instead).
    """
    with redshift_connect(settings) as conn:
        cur = conn.cursor()
        cur.execute(sql)
        df = cur.fetch_dataframe()
        if df is None:
            columns = [_column_name(c[0]).lower() for c in cur.description]
            df = pd.DataFrame(columns=columns)

    logger.info(f"Fetched {len(df)} rows")
    return df


def load_df_to_redshift(
    settings: Settings,
    df: pd.DataFrame,
    table: str,
    columns: Sequence[str],
    s3_file_name: str,
) -> int:
    """Append df to a Redshift table via S3 + COPY, then ANALYZE the table.

    Only `columns` are written, in that order, and COPY lists them
    explicitly, so DataFrame column order can never shift data between
    table columns. Returns the number of rows loaded (0 = nothing to load).
    """
    if df.empty:
        logger.info(f"No rows for {table}, skipping load")
        return 0

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame for {table} is missing columns: {missing}"
        )

    key = settings.s3_project_dir + settings.update_table_dir + s3_file_name
    s3_path = f"s3://{settings.bucket_name}/{key}"

    csv_buf = StringIO()
    df[list(columns)].to_csv(csv_buf, header=True, index=False)
    boto3.client("s3").put_object(
        Bucket=settings.bucket_name, Key=key, Body=csv_buf.getvalue()
    )
    logger.info(f"Uploaded {len(df)} rows to {s3_path}")

    copy_sql = (
        f"COPY {table} ({', '.join(columns)}) "
        f"FROM '{s3_path}' "
        f"IAM_ROLE '{settings.copy_role_arn}' "
        "CSV IGNOREHEADER 1;"
    )
    with redshift_connect(settings) as conn:
        cur = conn.cursor()
        cur.execute(copy_sql)
        conn.commit()
        cur.execute(f"ANALYZE {table}")
        conn.commit()

    logger.info(f"Loaded {len(df)} rows into {table}")
    return len(df)
