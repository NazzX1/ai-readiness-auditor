import pandas as pd

from file_handlers.readers.base_reader import BaseFileReader



class ParquetReader(BaseFileReader):
    def read(self):
        return pd.read_parquet(self.file_path)