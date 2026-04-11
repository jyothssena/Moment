"""
cloudsql_loader.py — MOMENT Cloud SQL Data Acquisition
=======================================================
Reads from Cloud SQL via Cloud SQL Python Connector.
Designed to run on Cloud Run (same VPC as Cloud SQL).
"""

import logging
import os
import struct
import yaml
import pandas as pd
import farmhash
from datetime import datetime

logger = logging.getLogger('moment_pipeline')


def make_user_id(character_name: str) -> int:
    raw    = farmhash.hash64(character_name)
    signed = struct.unpack('q', struct.pack('Q', raw))[0]
    return abs(signed)


DEFAULT_INSTANCE = os.environ.get('INSTANCE_CONNECTION_NAME', 'moment-486719:us-central1:moment-db')
DEFAULT_DB_NAME  = os.environ.get('CLOUDSQL_DB',   'momento')
DEFAULT_DB_USER  = os.environ.get('CLOUDSQL_USER',  'momento_admin')
DEFAULT_DB_PASS  = os.environ.get('CLOUDSQL_PASS',  '')


class CloudSQLLoader:

    def __init__(self, config_path=None):
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}

        self.timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dataframes = None

        csql = self.config.get("cloudsql", {})
        self.instance = csql.get("instance_connection_name") or DEFAULT_INSTANCE
        self.db_name  = csql.get("db_name") or DEFAULT_DB_NAME
        self.db_user  = csql.get("db_user") or DEFAULT_DB_USER
        self.db_pass  = csql.get("db_pass") or DEFAULT_DB_PASS

        self._engine = None
        logger.info(f"CloudSQLLoader initialised — instance={self.instance}, db={self.db_name}")

    def run(self):
        logger.info("Starting Cloud SQL data acquisition")
        self.dataframes = {
            "interpretations_train": self._load_interpretations(),
            "passage_details_new":   self._load_passages(),
            "user_details_new":      self._load_users(),
        }
        total_rows = sum(len(df) for df in self.dataframes.values())
        logger.info(f"Acquisition complete: {len(self.dataframes)} tables, {total_rows} total rows")
        return {
            "timestamp":  self.timestamp,
            "num_files":  len(self.dataframes),
            "total_rows": total_rows,
            "files":      list(self.dataframes.keys()),
            "source":     "cloudsql",
        }

    def get_dataframes(self):
        if self.dataframes is None:
            raise ValueError("No data loaded — call run() first.")
        return self.dataframes

    def _load_interpretations(self) -> pd.DataFrame:
        query = """
            SELECT
                m.id                                    AS interpretation_id,
                m.user_id                               AS user_id,
                m.user_id::text                         AS character_id,
                m.user_id::text                         AS character_name,
                m.book_id::text                         AS passage_id,
                b.title                                 AS book,
                b.author                                AS book_author,
                m.passage,
                m.chapter,
                m.page_num,
                m.interpretation,
                COALESCE(
                    array_length(
                        regexp_split_to_array(trim(m.interpretation), '\\s+'),
                        1
                    ), 0
                )                                       AS word_count,
                m.passage_key,
                m.created_at,
                m.interpretation_updated_at
            FROM public.moments m
            LEFT JOIN public.books b ON b.id = m.book_id
            WHERE m.is_deleted = FALSE
              AND m.interpretation IS NOT NULL
              AND trim(m.interpretation) <> ''
            ORDER BY m.created_at
        """
        logger.info("Loading interpretations from Cloud SQL")
        df = self._run_query(query)
        df['user_id']        = df['user_id'].astype(str).apply(make_user_id)
        df['character_id']   = df['user_id']
        df['character_name'] = df['user_id'].astype(str)
        logger.info(f"  -> {len(df)} rows")
        return df

    def _load_passages(self) -> pd.DataFrame:
        query = """
            SELECT
                b.id::text                  AS passage_id,
                b.title                     AS book_title,
                b.author                    AS book_author,
                b.year,
                b.gutenberg_id,
                b.cover_url,
                b.opening_passage,
                b.created_at,
                gc.epub_url,
                gc.text_url,
                gc.fetched_at               AS cache_fetched_at
            FROM public.books b
            LEFT JOIN public.gutenberg_book_cache gc ON gc.book_id = b.id
            ORDER BY b.title
        """
        logger.info("Loading passages from Cloud SQL")
        df = self._run_query(query)
        logger.info(f"  -> {len(df)} rows")
        return df

    def _load_users(self) -> pd.DataFrame:
        query = """
            SELECT
                m.user_id::text                         AS user_id,
                m.user_id::text                         AS Name,
                COUNT(*)                                AS moment_count,
                COUNT(DISTINCT m.book_id)               AS books_read,
                MIN(m.created_at)                       AS first_moment_at,
                MAX(m.created_at)                       AS last_moment_at,
                ROUND(AVG(COALESCE(
                    array_length(regexp_split_to_array(trim(m.interpretation), '\\s+'), 1), 0
                ))::numeric, 2)                         AS avg_word_count,
                COUNT(CASE WHEN m.interpretation IS NOT NULL
                      AND trim(m.interpretation) <> '' THEN 1 END) AS total_interpretations
            FROM public.moments m
            WHERE m.is_deleted = FALSE
            GROUP BY m.user_id
            ORDER BY moment_count DESC
        """
        logger.info("Loading users from Cloud SQL")
        df = self._run_query(query)
        df['user_id'] = df['user_id'].astype(str).apply(make_user_id)
        df['Name']    = df['user_id'].astype(str)
        logger.info(f"  -> {len(df)} rows (unique users)")
        return df

    def _get_engine(self):
        if self._engine is not None:
            return self._engine

        from google.cloud.sql.connector import Connector
        from sqlalchemy import create_engine

        connector = Connector()

        def _get_conn():
            return connector.connect(
                self.instance,
                "pg8000",
                user=self.db_user,
                password=self.db_pass,
                db=self.db_name,
                ip_type="PRIVATE"
            )

        self._engine = create_engine("postgresql+pg8000://", creator=_get_conn)
        logger.info("SQLAlchemy engine created via Cloud SQL Connector")
        return self._engine

    def _run_query(self, query: str) -> pd.DataFrame:
        try:
            engine = self._get_engine()
            with engine.connect() as conn:
                return pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"Cloud SQL query failed: {e}")
            raise