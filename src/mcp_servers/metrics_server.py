from mcp.server.fastmcp import FastMCP
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.tools import tool
from src.helpers.config import get_settings
from src.stores.RedisStore import RedisStore
from src.structured_metrics.metrics_calculator import (
    # Data Quality
    completeness_rate,
    uniqueness_rate,
    format_validity_rate,
    outlier_rate,
    total_missing_value_rate,
    schema_type_violations,
    exact_duplicate_overlap_rate,
    exact_duplicate_overlap_rate_across_splits,
    
    # ML & Statistical
    r_squared,
    variance_inflation_factor,
    population_stability_index,
    change_point_detection_score,
    
    # Class & Fairness
    class_imbalance_ratio,
    demographic_parity_disparity,
    
    # Target Leakage & PII
    target_leakage_count,
    potential_target_leakage_candidate_count,
    raw_pii_field_count,
    
    # PII & Privacy
    pii_contamination_rate,
    pii_phi_contamination_rate,
    pii_phiviolation_rate,
    
    # Image & Media
    aspect_ratio_variance,
    blur_index,
    iou_intersection_over_union,
    
    # Registry
    list_available_metrics,
)


settings = get_settings()


def create_store() -> RedisStore:
    return RedisStore(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=settings.REDIS_PASSWORD
    )

store = create_store()


mcp = FastMCP("AI Readiness Auditor - Metrics Server")

def fetch_dataframe_from_store(session_id: str, table_name: Optional[str] = None):
    return store.get_samples_as_dataframes(
        session_id=session_id, table_name=table_name
    )


@tool
def compute_completeness_tool(
    dataset_id: str,
    table_name: Optional[str] = None,
    column: Optional[str] = None,
) -> float:
    """Compute the completeness rate (non-null ratio) of a column or the entire dataset in Redis.

    Args:
        dataset_id: The session ID key stored in Redis.
        table_name: Target table/collection name (e.g., 'Album'). Optional if
          session has a single table.
        column: Target column name to evaluate. If None, computes average
          completeness for the full table.

    Returns:
        Completeness rate as a float between 0.0 and 1.0.
    """
    df = fetch_dataframe_from_store(dataset_id, table_name)
    return completeness_rate(df, column)


@tool
def compute_uniqueness_tool(
    dataset_id: str, column: str, table_name: Optional[str] = None
) -> float:
    """Compute the uniqueness rate for a given column.

    Args:
        dataset_id: The session ID key stored in Redis.
        column: Target column name to evaluate for unique values.
        table_name: Target table/collection name (e.g., 'Album').

    Returns:
        Uniqueness rate as a float between 0.0 and 1.0.
    """
    df = fetch_dataframe_from_store(dataset_id, table_name)
    return uniqueness_rate(df, column)


@tool
def compute_outliers_tool(
    dataset_id: str,
    column: str,
    table_name: Optional[str] = None,
    method: str = "iqr",
) -> float:
    """Compute the outlier rate for a given numeric column using IQR or Z-score.

    Args:
        dataset_id: The session ID key stored in Redis.
        column: Target numeric column name to analyze.
        table_name: Target table/collection name (e.g., 'Album').
        method: Detection strategy to use ('iqr' or 'zscore'). Defaults to
          'iqr'.

    Returns:
        Ratio of identified outlier records to total rows (0.0 to 1.0).
    """
    df = fetch_dataframe_from_store(dataset_id, table_name)
    return outlier_rate(df, column, method=method)


@tool
def check_format_validity_tool(
    dataset_id: str,
    column: str,
    table_name: Optional[str] = None,
    pattern: Optional[str] = None,
) -> float:
    """Check format validity rate for values in a column matching a regex pattern.

    Args:
        dataset_id: The session ID key stored in Redis.
        column: Target text column name to inspect.
        table_name: Target table/collection name (e.g., 'Album').
        pattern: Regex pattern string to validate entries against.

    Returns:
        Ratio of matching valid values to total entries (0.0 to 1.0).
    """
    df = fetch_dataframe_from_store(dataset_id, table_name)
    return format_validity_rate(df, column, pattern=pattern)


@tool
def check_schema_violations_tool(
    dataset_id: str,
    table_name: Optional[str] = None,
    expected_types: Optional[Dict[str, str]] = None,
) -> int:
    """Check for schema data type violations across specified columns.

    Args:
        dataset_id: The session ID key stored in Redis.
        table_name: Target table/collection name (e.g., 'Album').
        expected_types: Mapping of column names to expected type strings (e.g.,
          {'AlbumId': 'int', 'Title': 'str'}).

    Returns:
        Count of total schema violation instances detected.
    """
    df = fetch_dataframe_from_store(dataset_id, table_name)
    return schema_type_violations(df, expected_types or {})


@tool
def compute_r_squared_tool(
    y_true: List[float],
    y_pred: List[float],
) -> Optional[float]:
    """Compute R^2 determination score between true and predicted targets.

    Args:
        y_true: List of ground truth float target values.
        y_pred: List of predicted float target values.

    Returns:
        R^2 score as float, or None if input lists are empty.
    """
    if not y_true or not y_pred:
        return None
    return float(
        r_squared(np.array(y_true, dtype=float), np.array(y_pred, dtype=float))
    )


@tool
def compute_vif_tool(
    dataset_id: str,
    column: str,
    table_name: Optional[str] = None,
    other_columns: Optional[List[str]] = None,
) -> float:
    """Compute Variance Inflation Factor (VIF) for multicollinearity check.

    Args:
        dataset_id: The session ID key stored in Redis.
        column: Feature column name to calculate VIF for.
        table_name: Target table/collection name (e.g., 'Album').
        other_columns: Optional list of predictor feature columns to include.

    Returns:
        VIF numeric value (values > 5.0 indicating high multicollinearity).
    """
    df = fetch_dataframe_from_store(dataset_id, table_name)
    return variance_inflation_factor(df, column, other_columns)


@tool
def detect_population_shift_tool(
    expected: List[float],
    actual: List[float],
    bins: int = 10,
) -> Optional[float]:
    """Calculate Population Stability Index (PSI) to detect distribution drift.

    Args:
        expected: List of float values representing baseline/training
          distribution.
        actual: List of float values representing current/target distribution.
        bins: Number of equal-sized buckets to discretize distributions into.
          Defaults to 10.

    Returns:
        PSI score as float (>0.2 indicates significant shift), or None if inputs
        empty.
    """
    if not expected or not actual:
        return None
    return float(
        population_stability_index(
            np.array(expected, dtype=float),
            np.array(actual, dtype=float),
            bins,
        )
    )


@tool
def detect_change_points_tool(timeseries: List[float]) -> Optional[float]:
    """Detect change points in a given numeric time series.

    Args:
        timeseries: Ordered sequence of float metric values over time.

    Returns:
        Change point detection confidence score as a float, or None if input
        empty.
    """
    if not timeseries:
        return None
    return float(
        change_point_detection_score(np.array(timeseries, dtype=float))
    )


@tool
def compute_class_imbalance_tool(
    dataset_id: str, target_column: str, table_name: Optional[str] = None
) -> float:
    """Compute minority to majority class ratio for a classification target column.

    Args:
        dataset_id: The session ID key stored in Redis.
        target_column: Target categorical class column name.
        table_name: Target table/collection name (e.g., 'Album').

    Returns:
        Ratio of minority count to majority count (0.0 to 1.0).
    """
    df = fetch_dataframe_from_store(dataset_id, table_name)
    return class_imbalance_ratio(df, target_column)


@tool
def measure_demographic_parity_tool(
    dataset_id: str,
    protected_attr: str,
    outcome_attr: str,
    table_name: Optional[str] = None,
    positive_label: int = 1,
) -> float:
    """Measure demographic parity disparity for algorithmic fairness checks.

    Args:
        dataset_id: The session ID key stored in Redis.
        protected_attr: Column name for sensitive attribute (e.g., 'gender',
          'age_group').
        outcome_attr: Column name for decision outcome/prediction (e.g.,
          'approved').
        table_name: Target table/collection name (e.g., 'Users').
        positive_label: Value signifying positive decision outcome. Defaults to
          1.

    Returns:
        Disparity delta between privileged and unprivileged group positive rates.
    """
    df = fetch_dataframe_from_store(dataset_id, table_name)
    return demographic_parity_disparity(
        df, protected_attr, outcome_attr, positive_label
    )


@tool
def detect_target_leakage_tool(
    dataset_id: str,
    target_column: str,
    table_name: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
) -> int:
    """Detect candidate target leakage features based on suspicious correlations.

    Args:
        dataset_id: The session ID key stored in Redis.
        target_column: Target model prediction column name.
        table_name: Target table/collection name.
        feature_columns: Optional subset of feature columns to inspect against
          target.

    Returns:
        Count of feature columns showing high target leakage suspicion.
    """
    df = fetch_dataframe_from_store(dataset_id, table_name)
    return target_leakage_count(df, target_column, feature_columns)


@tool
def count_pii_fields_tool(
    dataset_id: str, table_name: Optional[str] = None
) -> int:
    """Count raw PII/sensitive identifier candidate fields in the dataset.

    Args:
        dataset_id: The session ID key stored in Redis.
        table_name: Target table/collection name.

    Returns:
        Number of detected sensitive/PII column names.
    """
    df = fetch_dataframe_from_store(dataset_id, table_name)
    return raw_pii_field_count(df)


@tool
def measure_pii_contamination_tool(
    dataset_id: str,
    table_name: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> float:
    """Measure PII record contamination rate across specified or all text columns.

    Args:
        dataset_id: The session ID key stored in Redis.
        table_name: Target table/collection name.
        columns: Specific text column names to scan. If None, scans all string
          columns.

    Returns:
        Fraction of rows containing detected PII pattern instances (0.0 to 1.0).
    """
    df = fetch_dataframe_from_store(dataset_id, table_name)
    return pii_contamination_rate(df, columns)


@tool
def measure_phi_contamination_tool(
    dataset_id: str,
    table_name: Optional[str] = None,
    columns: Optional[List[str]] = None,
) -> float:
    """Measure PHI contamination rate for healthcare and privacy compliance.

    Args:
        dataset_id: The session ID key stored in Redis.
        table_name: Target table/collection name.
        columns: Specific columns to analyze for PHI entities (e.g. medical IDs,
          diagnoses).

    Returns:
        Ratio of records containing unmasked PHI content (0.0 to 1.0).
    """
    df = fetch_dataframe_from_store(dataset_id, table_name)
    return pii_phi_contamination_rate(df, columns)


@tool
def measure_phi_violations_tool(
    dataset_id: str,
    table_name: Optional[str] = None,
    regulated_fields: Optional[List[str]] = None,
) -> float:
    """Measure PHI compliance violation rate on regulated sensitive fields.

    Args:
        dataset_id: The session ID key stored in Redis.
        table_name: Target table/collection name.
        regulated_fields: Specific HIPAA/regulated field names to evaluate.

    Returns:
        Compliance violation rate as a float between 0.0 and 1.0.
    """
    df = fetch_dataframe_from_store(dataset_id, table_name)
    return pii_phiviolation_rate(df, regulated_fields)


@tool
def compute_aspect_ratio_variance_tool(
    image_dimensions: List[List[int]],
) -> float:
    """Compute aspect ratio variance across image bounding boxes or frames.

    Args:
        image_dimensions: List of [width, height] pairs for image assets (e.g.,
          [[1920, 1080], [800, 600]]).

    Returns:
        Variance value of computed width-to-height aspect ratios.
    """
    return aspect_ratio_variance(image_dimensions or [])


@tool
def compute_blur_index_tool(image_array: List[List[float]]) -> float:
    """Calculate image blur index using Laplacian variance.

    Args:
        image_array: 2D matrix of grayscale pixel float values.

    Returns:
        Laplacian variance score (lower values indicate higher image blur).
    """
    if not image_array:
        return 0.0
    return float(blur_index(np.array(image_array, dtype=np.uint8)))


@tool
def compute_iou_tool(
    box1: List[int],
    box2: List[int],
) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First bounding box formatted as [x1, y1, x2, y2].
        box2: Second bounding box formatted as [x1, y1, x2, y2].

    Returns:
        IoU score as a float value between 0.0 and 1.0.
    """
    b1 = tuple(box1) if box1 and len(box1) == 4 else (0, 0, 0, 0)
    b2 = tuple(box2) if box2 and len(box2) == 4 else (0, 0, 0, 0)
    return float(iou_intersection_over_union(b1, b2))


@tool
def list_metrics() -> List[str]:
    """List all available metrics in the registry.

        Returns:
            List of registered metric name strings.
    """
    return list_available_metrics()


@tool
def get_metrics_info() -> Dict[str, str]:
    """Get information about available metric categories.

        Returns:
            Dictionary mapping domain category keys to descriptive metric summaries.
    """
    return {
        "data_quality": (
            "completeness, uniqueness, format validity, outliers, duplicates"
        ),
        "ml_statistical": "R-squared, VIF, PSI, change point detection",
        "fairness": "class imbalance, demographic parity",
        "leakage": "target leakage detection, PII fields",
        "privacy": "PII contamination, PHI contamination",
        "media": "aspect ratio variance, IoU",
    }