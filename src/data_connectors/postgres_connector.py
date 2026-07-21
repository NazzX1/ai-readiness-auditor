from src.data_connectors.base_connector import BaseConnector
from src.models.schema_models import ColumnInfo, TableMetadata
from sqlalchemy import text, create_engine, inspect


class PostgresConnector(BaseConnector):
    def __init__(self, uri : str):
        path = uri.split("://")[-1]
        if "/" not in path:
            raise ValueError("invalid uri format. expected format: postgresql://user:password@host:port/db_name")
        if not path.split("/")[-1]:
            raise ValueError("database name is missing in the uri. expected format: postgresql://user:password@host:port/db_name")
        
        #self.base_url = f"postgresql://{user}:{password}@{host}:{port}/{db_name}"
        self.engine = create_engine(uri)
        self.inspector = inspect(self.engine)


    def get_all_tables_or_collections(self, schema = "public"):
        return self.inspector.get_table_names(schema=schema)
    

    def get_full_metadata(self, schema = "public"):
        
        schema = {}
        tables = self.inspector.get_table_names(schema=None)

        for table in tables:
            schema[table] = self.get_table_metadata(table_name=table)
        
        return schema
        
        
    
    def get_table_metadata(self, table_name, schema = "public"):
        
        foreign_keys = self.inspector.get_foreign_keys(table_name=table_name, schema=schema)

        cols_res = self.inspector.get_columns(table_name=table_name, schema=schema)
        columns = [ColumnInfo(c.get("name"), c.get("type"), c.get("nullable"), c.get("comment", "")) for c in cols_res]

        stats = self._get_table_stats(table_name=table_name)

        return TableMetadata(table_name=table_name,
                             columns=columns,
                             foreign_keys=foreign_keys,
                             row_count=stats["row_count"],
                             size_bytes=stats["size_bytes"])



    def get_sample_from_table(self, table_name : str, percentage : float = 0.1):
        query = text(f'select * from "{table_name}" TABLESAMPLE SYSTEM(:p)')
        with self.engine.connect() as conn:
            results = conn.execute(query, {"p": percentage}).fetchall()
            return [dict(row) for row in results]
        

    def get_sample(self):
        tables = self.get_all_tables_or_collections()
        samples = {}
        for table in tables:
            samples[table] = self.get_sample_from_table(table)
        return samples
    


    def _get_table_stats(self, table_name : str, schema : str = "public"):
        query = text("""
                    SELECT reltuples::bigint as row_count, 
                    pg_total_relation_size(quote_ident(:schema) || '.' || quote_ident(:table)) AS size_bytes
                    FROM pg_class c
                    JOIN pg_namespace n on n.oid = c.relnamespace
                    where n.nspname = :schema AND c.relname = :table
                    """)
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {"schema" : schema, "table": table_name}).fetchone()
            return {
                "row_count" : result.row_count if result else 0,
                "size_bytes" : result.size_bytes if result else 0
            }