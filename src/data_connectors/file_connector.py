from src.file_handlers.readers import CSVReader, JSONReader, ParquetReader, ExcelReader
import os
from src.data_connectors.base_connector import BaseConnector
from src.models.schema_models import ColumnInfo, TableMetadata

class FileConnector(BaseConnector):
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_type = self._determine_file_type()
        self.reader = self._get_reader()

    def _determine_file_type(self):
        if self.file_path.endswith('.csv'):
            return 'csv'
        elif self.file_path.endswith('.json'):
            return 'json'
        elif self.file_path.endswith('.parquet'):
            return 'parquet'
        elif self.file_path.endswith('.xlsx') or self.file_path.endswith('.xls'):
            return 'excel'
        else:
            raise ValueError(f"Unsupported file type for {self.file_path}")

    def _get_reader(self):
        READER_MAP = {
            "csv": CSVReader,
            "json": JSONReader,
            "parquet": ParquetReader,
            "excel": ExcelReader
        }

        if self.file_type in READER_MAP:
            return READER_MAP[self.file_type](self.file_path)
        else:
            raise ValueError(f"No reader available for this type: {self.file_type}")

    def read(self):
        return self.reader.read()
    

    def get_all_tables_or_collections(self):
        return [os.path.basename(self.file_path)]

    def get_full_metadata(self, table_name : str = ""):
        keys = self.reader.parse()
        return {os.path.basename(self.file_path) : TableMetadata(
            table_name=os.path.basename(self.file_path),
            columns=[ColumnInfo(name=key, data_type="Unknown", nullable=True) for key in keys],
            foreign_keys=[],
            row_count=len(self.reader.read()),
            size_bytes=os.path.getsize(self.file_path)
        )}


    def get_sample(self, table_name : str = "", limit : int = 10):
        return self.reader.read().head(limit)
    

    