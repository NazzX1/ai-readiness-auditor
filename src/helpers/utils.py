import pandas as pd
import json
from decimal import Decimal
from datetime import datetime, date, time
import uuid
import base64

class PostgresMetadataEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj) if obj % 1 != 0 else int(obj)
            
        if isinstance(obj, (datetime, date, time)):
            return obj.isoformat()
            
        if isinstance(obj, uuid.UUID):
            return str(obj)

        elif isinstance(obj, memoryview):
            return base64.b64encode(obj.tobytes()).decode("utf-8")
            
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='ignore')
            
        if hasattr(obj, '__str__'):
            return str(obj)

        return super().default(obj)



def flatten_json(data):
    rows = []

    def recurse(obj, path=[]):
        if isinstance(obj, dict):
            for key, val in obj.items():
                recurse(val, path + [key])
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    row = item.copy()
                    rows.append(row)

    recurse(data)
    return pd.DataFrame(rows)


def serialize_value(val):
        if isinstance(val, Decimal):
            return float(val) if val % 1 != 0 else int(val)
        if isinstance(val, (datetime, date, time)):
            return val.isoformat()
        if isinstance(val, uuid.UUID):
            return str(val)
        if isinstance(val, bytes):
            return val.decode("utf-8", errors="ignore")
        return val