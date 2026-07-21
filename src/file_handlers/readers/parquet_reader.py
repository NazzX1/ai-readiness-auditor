import pandas as pd

from src.file_handlers.readers.base_reader import BaseFileReader



class ParquetReader(BaseFileReader):
    def read(self):
        return pd.read_parquet(self.file_path)
    def parse(self):
        return dict(zip(pd.read_parquet(self.file_path, engine='pyarrow', nrows=0).columns, list(pd.read_parquet(self.file_path, engine='pyarrow', nrows=0).dtypes.astype(str))))