from dataclasses import dataclass, Field
from typing import List, Any, Optional
from enum import Enum

@dataclass
class ColumnInfo:
    name : str
    data_type : str
    is_nullable : bool
    comment : Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "data_type": str(self.data_type),
            "is_nullable": self.is_nullable,
            "comment": self.comment
        }



@dataclass
class TableMetadata:
    table_name : str
    columns : List[ColumnInfo]
    foreign_keys : List[Any]
    row_count : int
    size_bytes : int



@dataclass
class Metric:
    name : str
    description : Optional[str]
    formula : str
    pass_threshold : Optional[float] = None
    fail_threshold : Optional[float] = None
    warn_threshold : Optional[float] = None

class Scope(Enum):
    AI = "AI"
    ML = "ML"


class SensitivityLevel(Enum):
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    PII = "PII"
    RESTRICTED = "RESTRICTED"

class TaskCategory(Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTI_CLASS_CLASSIFICATION = "multi_class_classification"
    MULTI_LABEL_CLASSIFICATION = "multi_label_classification"
    
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    ANOMALY_DETECTION = "anomaly_detection"
    
    FORECASTING = "forecasting"
    RANKING = "ranking"
    RECOMMENDATION = "recommendation"
    GENERATIVE = "generative"