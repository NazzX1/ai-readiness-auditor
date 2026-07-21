import os
import pandas as pd
import openpyxl
from file_handlers.readers.base_reader import BaseFileReader





_XL_EXTENSIONS = (".xlsx", ".xlsm")

class ExcelReader(BaseFileReader):
    def read(self):
        extension = os.path.splitext(self.file_path)[1].lower()
        if extension in _XL_EXTENSIONS:
            return pd.read_excel(self.file_path)
    

    def parse(self):
        extension = os.path.splitext(self.file_path)[1].lower()
        if extension in _XL_EXTENSIONS:
            return list(pd.read_excel(self.file_path, nrows=0).columns)