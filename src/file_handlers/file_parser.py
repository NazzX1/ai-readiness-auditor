from src.file_handlers.readers import CSVReader, JSONReader, ParquetReader, ExcelReader
import os



READER_MAP = {
    "csv" : CSVReader,
    "json" : JSONReader,
    "parquet" : ParquetReader,
    "excel" : ExcelReader
}


SUPPORTED_FILE_TYPES = [
    (".csv", "CSV"),
    (".xlsx, .xlsm", "Excel"),
    (".json", "JSON"),
    (".parquet", "Parquet"),
]




def parse_file(file_info):
    
    file_path, file_name, file_type = file_info[:3]
    try:
        if file_type in READER_MAP:
            keys = READER_MAP[file_type](file_path).parse()
            return keys
        else:
            print(f"Unsupported file type: {file_type}")
            return None
    except Exception as e:
        print(f"Error while parsing: {e}")
        return str(e)



def filter_file(file_info, kept_keys):
    file_path, _, file_type = file_info[:3]
    if file_type in READER_MAP:
        filtered_file_path = READER_MAP[file_type](file_path).filter(kept_keys)
        print(f"Filtered file saved to: {filtered_file_path}")
        return filtered_file_path
    else:
        print(f"Unsupported file type: {file_type}")
        return None




def read_file(file_info, columns=None):


    file_path, file_name, file_type = file_info[:3]
    
    if file_path and not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return f"File not found: {file_path}"

    if file_type not in READER_MAP:
        print(f"Unsupported file type: {file_type}")
        return None


    try:
        data = READER_MAP[file_type](file_path).read()
        return data[columns]
    except Exception as e:
        print(f"Unable to read the file: {e}")
        return str(e)
