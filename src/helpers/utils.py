import pandas as pd


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