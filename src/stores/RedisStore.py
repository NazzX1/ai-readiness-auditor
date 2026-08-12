import json
from typing import List, Dict, Any, Optional, Tuple
import redis
import pandas as pd
from src.helpers.utils import PostgresMetadataEncoder

class RedisStore:

    def __init__(
        self, 
        host: str , 
        port: int , 
        db: int = 0, 
        password: Optional[str] = None,
        default_ttl_seconds: int = 3600
    ):
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True 
        )
        self.default_ttl = default_ttl_seconds

    def _get_key(self, session_id: str) -> str:
        return f"session:{session_id}:samples"

    def save_samples(
        self, 
        session_id: str, 
        samples: List[Dict[str, Any]], 
        ttl_seconds: Optional[int] = None
    ) -> bool:

        key = self._get_key(session_id)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl

        payload = json.dumps(samples, cls = PostgresMetadataEncoder)

        return self.client.setex(name=key, time=ttl, value=payload)

    def get_samples(self, session_id: str) -> Optional[List[Dict[str, Any]]]:

        key = self._get_key(session_id)
        data = self.client.get(key)
        
        if not data:
            return None

        return json.loads(data)

    def get_samples_as_dataframes(self, session_id: str, table_name: Optional[str] = None) :

        raw_samples = self.get_samples(session_id)

        if not raw_samples:
            return None

        tables_dict = (
            raw_samples[0] if isinstance(raw_samples, list) else raw_samples
        )

        def _convert_to_df(table_key: str, records: list) -> pd.DataFrame:
            df = pd.json_normalize(records)

            primary_id_col = f"{table_key}Id"
            if primary_id_col in df.columns:
                df = df.set_index(primary_id_col)

            return df

        if table_name:
            if table_name not in tables_dict:
                raise ValueError(
                    f"Table '{table_name}' not found in session '{session_id}'"
                )
            return _convert_to_df(table_name, tables_dict[table_name])

        return {
            table: _convert_to_df(table, records)
            for table, records in tables_dict.items()
        }

    def get_col_sample(self, session_id: str, table_name : str|None, col_name : str):
        return self.get_samples_as_dataframes(session_id=session_id, table_name=table_name)[col_name]
    

    def get_table_as_df(self, session_id : str, table_name : str):
        return self.get_samples_as_dataframes(session_id=session_id, table_name=table_name)


    def get_samples_paginated(
        self,
        session_id: str,
        table_name: str,
        page: int = 1,
        page_size: int = 10,
    ) -> Dict[str, Any]:
        if page < 1 or page_size < 1:
            raise ValueError("Page and page_size must be 1 or greater.")

        all_samples = self.get_samples(session_id)

        if all_samples is None or table_name not in all_samples:
            return {
                "session_id": session_id,
                "table_name": table_name,
                "status": "expired_or_not_found",
                "total_records": 0,
                "page": page,
                "page_size": page_size,
                "total_pages": 0,
                "data": [],
            }

        records = all_samples[table_name]
        total_records = len(records)
        total_pages = (total_records + page_size - 1) // page_size

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        return {
            "session_id": session_id,
            "table_name": table_name,
            "status": "success",
            "total_records": total_records,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "data": records[start_idx:end_idx],
        }
    def extend_ttl(self, session_id: str, ttl_seconds: Optional[int] = None) -> bool:
        key = self._get_key(session_id)
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        return self.client.expire(key, ttl)

    def delete_session(self, session_id: str) -> bool:
        key = self._get_key(session_id)
        return bool(self.client.delete(key))