from src.file_handlers.readers.base_reader import BaseFileReader 
import pandas as pd


class CSVReader(BaseFileReader):
    def read(self):
        return pd.read_csv(self.file_path, index_col=False)
    
    def parse(self):
        return dict(zip(pd.read_csv(self.file_path, nrows=0).columns, list(str(i) for i in pd.read_csv(self.file_path, nrows=1).dtypes)))