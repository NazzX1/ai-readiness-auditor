import os
import json
import uuid
from file_handlers.readers.base_reader import BaseFileReader
from helpers.utils import flatten_json
from helpers.config import get_settings
from pathlib import Path




settings = get_settings()

class JSONReader(BaseFileReader):
    def read(self):
        with open(self.file_path) as f:
            data = json.load(f)

        return flatten_json(data)
    


    def parse(self):
        with open(self.file_path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            keys = [str(key) for key in data.keys()]
            return keys


    def filter(self, kept_keys):
        if isinstance(kept_keys, str):
            kept_keys = kept_keys.split(",")
        
        kept_keys = [str(key).strip() for key in kept_keys]
        
        with open(self.file_path) as f:
            data = json.load(f)
        

        filtered_data = {key: data[key] for key in kept_keys if key in data}

        new_filename = f"filtered_{Path(self.file_path).name}_{uuid.uuid4().hex}"
        new_file_path = os.path.join(settings.UPLOADED_DATA, new_filename) 

        with open(new_file_path, "w") as f:
            json.dump(filtered_data, f, indent=2)
            
        return new_file_path
                    