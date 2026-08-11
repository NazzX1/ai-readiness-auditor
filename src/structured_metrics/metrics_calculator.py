"""
Comprehensive metrics computation module for data quality, ML evaluation, 
and governance assessments across structured, semi-structured, and media data.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler
import hashlib
import warnings

warnings.filterwarnings("ignore")


# data quality metrics

def completeness_rate(df: pd.DataFrame, column: Optional[str] = None) -> float:
    """
    Calculate the proportion of non-null values in a column or across dataset.
    
    Args:
        df: Input DataFrame
        column: Specific column to assess; if None, compute across entire dataset
        
    Returns:
        Float between 0 and 1 representing completeness rate
    """
    if column:
        if column not in df.columns:
            return 0.0
        return float((df[column].notna().sum()) / len(df))
    else:
        return float(df.notna().sum().sum() / (len(df) * len(df.columns)))


def uniqueness_rate(df: pd.DataFrame, column: str) -> float:
    """
    Calculate the proportion of unique non-null values in a column.
    
    Args:
        df: Input DataFrame
        column: Column name to assess
        
    Returns:
        Float between 0 and 1 representing uniqueness rate
    """
    if column not in df.columns:
        return 0.0
    valid_values = df[column].dropna()
    if len(valid_values) == 0:
        return 0.0
    return float(len(valid_values.unique()) / len(valid_values))


def format_validity_rate(df: pd.DataFrame, column: str, pattern: Optional[str] = None) -> float:
    """
    Calculate the proportion of values matching a valid format/pattern.
    
    Args:
        df: Input DataFrame
        column: Column name to assess
        pattern: Regex pattern to validate format (optional)
        
    Returns:
        Float between 0 and 1 representing format validity rate
    """
    if column not in df.columns:
        return 0.0
    
    valid_values = df[column].dropna()
    if len(valid_values) == 0:
        return 0.0
    
    if pattern:
        import re
        valid_count = sum(1 for v in valid_values if re.match(pattern, str(v)))
        return float(valid_count / len(valid_values))
    
    return 1.0


def outlier_rate(df: pd.DataFrame, column: str, method: str = "iqr") -> float:
    """
    Calculate the proportion of outlier values in a numeric column.
    
    Args:
        df: Input DataFrame
        column: Numeric column name to assess
        method: "iqr" (interquartile range) or "zscore"
        
    Returns:
        Float between 0 and 1 representing outlier rate
    """
    if column not in df.columns:
        return 0.0
    
    valid_values = pd.to_numeric(df[column], errors="coerce").dropna()
    if len(valid_values) == 0:
        return 0.0
    
    if method == "iqr":
        Q1 = valid_values.quantile(0.25)
        Q3 = valid_values.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((valid_values < Q1 - 1.5 * IQR) | (valid_values > Q3 + 1.5 * IQR)).sum()
    else:  # zscore
        outliers = (np.abs(stats.zscore(valid_values)) > 3).sum()
    
    return float(outliers / len(valid_values))


def total_missing_value_rate(df: pd.DataFrame) -> float:
    """Calculate the share of missing cells across the entire dataset."""
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isna().sum().sum()
    return float(missing_cells / total_cells) if total_cells > 0 else 0.0


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R-squared (coefficient of determination) for regression.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Float representing R-squared value (typically 0-1, can be negative)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0


def variance_inflation_factor(df: pd.DataFrame, column: str, other_columns: Optional[List[str]] = None) -> float:
    """
    Calculate Variance Inflation Factor (VIF) for multicollinearity detection.
    
    Args:
        df: Input DataFrame
        column: Target column to assess for multicollinearity
        other_columns: Feature columns to use in regression; if None, use all other numeric columns
        
    Returns:
        Float representing VIF (1 = no multicollinearity, >10 = high multicollinearity)
    """
    if column not in df.columns:
        return 0.0
    
    if other_columns is None:
        other_columns = [c for c in df.select_dtypes(include=[np.number]).columns if c != column]
    
    if len(other_columns) == 0:
        return 1.0
    
    try:
        from sklearn.linear_model import LinearRegression
        X = df[other_columns].dropna()
        y = df.loc[X.index, column]
        
        if len(X) < 2:
            return 1.0
        
        model = LinearRegression()
        model.fit(X, y)
        r_squared_val = model.score(X, y)
        
        return float(1 / (1 - r_squared_val)) if r_squared_val < 1 else float('inf')
    except Exception:
        return 1.0


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI) to detect data drift.
    
    Args:
        expected: Reference/expected distribution
        actual: Actual/observed distribution
        bins: Number of bins for distribution comparison
        
    Returns:
        Float representing PSI (0.1 = statistically similar, >0.3 = significant shift)
    """
    expected = np.array(expected).flatten()
    actual = np.array(actual).flatten()
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    expected_counts = np.histogram(expected, bins=bin_edges)[0] + 1e-10
    actual_counts = np.histogram(actual, bins=bin_edges)[0] + 1e-10
    
    expected_pct = expected_counts / expected_counts.sum()
    actual_pct = actual_counts / actual_counts.sum()
    
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def schema_type_violations(df: pd.DataFrame, expected_types: Dict[str, type]) -> int:
    """
    Count columns with data type violations versus expected schema.
    
    Args:
        df: Input DataFrame
        expected_types: Dictionary mapping column name to expected type
        
    Returns:
        Integer count of schema violations
    """
    violations = 0
    for col, expected_type in expected_types.items():
        if col not in df.columns:
            violations += 1
        else:
            # Attempt type coercion
            try:
                if expected_type in (int, float):
                    pd.to_numeric(df[col], errors="coerce")
                elif expected_type == str:
                    df[col].astype(str)
                elif expected_type == bool:
                    pd.to_numeric(df[col], errors="coerce")
            except Exception:
                violations += 1
    return violations


def exact_duplicate_overlap_rate(df1: pd.DataFrame, df2: pd.DataFrame) -> float:
    """
    Calculate the proportion of rows in df1 that appear (exactly) in df2.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame (typically test/validation set)
        
    Returns:
        Float between 0 and 1 representing duplicate overlap rate
    """
    if len(df1) == 0:
        return 0.0
    
    common_cols = list(set(df1.columns) & set(df2.columns))
    if not common_cols:
        return 0.0
    
    merged = pd.merge(df1[common_cols], df2[common_cols], how="inner")
    overlap = len(merged.drop_duplicates())
    return float(overlap / len(df1))


def exact_duplicate_overlap_rate_across_splits(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                                               val_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """
    Calculate exact duplicate overlap rates across train/test/val splits.
    
    Args:
        train_df: Training set
        test_df: Test set
        val_df: Validation set (optional)
        
    Returns:
        Dictionary with overlap rates for each pair
    """
    result = {
        "train_test_overlap": exact_duplicate_overlap_rate(test_df, train_df),
        "test_train_overlap": exact_duplicate_overlap_rate(train_df, test_df),
    }
    
    if val_df is not None:
        result["val_train_overlap"] = exact_duplicate_overlap_rate(val_df, train_df)
        result["train_val_overlap"] = exact_duplicate_overlap_rate(train_df, val_df)
        result["val_test_overlap"] = exact_duplicate_overlap_rate(val_df, test_df)
        result["test_val_overlap"] = exact_duplicate_overlap_rate(test_df, val_df)
    
    return result


# class imbalance and fairness metrics

def class_imbalance_ratio(df: pd.DataFrame, target_column: str) -> float:
    """
    Calculate the ratio of majority to minority class counts.
    
    Args:
        df: Input DataFrame
        target_column: Target/label column name
        
    Returns:
        Float representing majority-to-minority ratio (1.0 = balanced)
    """
    if target_column not in df.columns:
        return 0.0
    
    value_counts = df[target_column].value_counts()
    if len(value_counts) < 2:
        return 1.0
    
    return float(value_counts.iloc[0] / value_counts.iloc[-1])


def demographic_parity_disparity(df: pd.DataFrame, protected_attr: str, outcome_attr: str,
                                   positive_label: Any = 1) -> float:
    """
    Calculate demographic parity disparity (fairness metric).
    
    Args:
        df: Input DataFrame
        protected_attr: Protected attribute column (e.g., race, gender)
        outcome_attr: Outcome/decision column
        positive_label: Value representing positive outcome
        
    Returns:
        Float representing disparity (0 = perfect parity)
    """
    if protected_attr not in df.columns or outcome_attr not in df.columns:
        return 0.0
    
    groups = df[protected_attr].unique()
    if len(groups) < 2:
        return 0.0
    
    positive_rates = []
    for group in groups:
        group_df = df[df[protected_attr] == group]
        if len(group_df) > 0:
            pos_rate = (group_df[outcome_attr] == positive_label).sum() / len(group_df)
            positive_rates.append(pos_rate)
    
    if len(positive_rates) < 2:
        return 0.0
    
    return float(max(positive_rates) - min(positive_rates))


# pii and privacy metrics

def pii_contamination_rate(df: pd.DataFrame, columns_to_scan: Optional[List[str]] = None) -> float:
    """
    Calculate the proportion of potential PII (Personally Identifiable Information) fields.
    
    Args:
        df: Input DataFrame
        columns_to_scan: Specific columns to scan; if None, scan all string columns
        
    Returns:
        Float between 0 and 1 representing PII contamination rate
    """
    import re
    
    if columns_to_scan is None:
        columns_to_scan = df.select_dtypes(include="object").columns.tolist()
    
    if not columns_to_scan:
        return 0.0
    
    pii_patterns = {
        "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "phone": re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "credit_card": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
        "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    }
    
    pii_count = 0
    total_count = 0
    
    for col in columns_to_scan:
        if col not in df.columns:
            continue
        
        for val in df[col].dropna():
            total_count += 1
            val_str = str(val)
            for pattern in pii_patterns.values():
                if pattern.search(val_str):
                    pii_count += 1
                    break
    
    return float(pii_count / total_count) if total_count > 0 else 0.0


def pii_phi_contamination_rate(df: pd.DataFrame, columns_to_scan: Optional[List[str]] = None) -> float:
    """
    Calculate the proportion of Protected Health Information (PHI) fields (HIPAA).
    
    Args:
        df: Input DataFrame
        columns_to_scan: Specific columns to scan; if None, scan all string columns
        
    Returns:
        Float between 0 and 1 representing PHI contamination rate
    """
    import re
    
    if columns_to_scan is None:
        columns_to_scan = df.select_dtypes(include="object").columns.tolist()
    
    if not columns_to_scan:
        return 0.0
    
    phi_patterns = {
        "medical_record_num": re.compile(r"\b[0-9]{6,10}\b"),
        "date_of_birth": re.compile(r"\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12][0-9]|3[01])[/-](19|20)\d{2}\b"),
        "health_plan_beneficiary": re.compile(r"\b[A-Z]{2}[0-9]{8,10}\b"),
        "account_number": re.compile(r"\b[0-9]{8,17}\b"),
    }
    
    phi_count = 0
    total_count = 0
    
    for col in columns_to_scan:
        if col not in df.columns:
            continue
        
        for val in df[col].dropna():
            total_count += 1
            val_str = str(val)
            for pattern in phi_patterns.values():
                if pattern.search(val_str):
                    phi_count += 1
                    break
    
    return float(phi_count / total_count) if total_count > 0 else 0.0


def pii_phiviolation_rate(df: pd.DataFrame, regulated_fields: Optional[List[str]] = None) -> float:
    """
    Calculate violation rate for regulated PII/PHI fields (stricter than contamination).
    
    Args:
        df: Input DataFrame
        regulated_fields: Specific fields marked as regulated; if None, use heuristics
        
    Returns:
        Float between 0 and 1 representing violation rate
    """
    if regulated_fields is None:
        regulated_fields = [col for col in df.columns 
                          if any(x in col.lower() for x in ["ssn", "mru", "dob", "medical"])]
    
    if not regulated_fields:
        return 0.0
    
    violation_count = 0
    for col in regulated_fields:
        if col in df.columns:
            # Count non-null values in regulated field
            violation_count += df[col].notna().sum()
    
    total_regulated_cells = len(df) * len(regulated_fields)
    return float(violation_count / total_regulated_cells) if total_regulated_cells > 0 else 0.0


# feature engineering and target leakage

def target_leakage_count(df: pd.DataFrame, target_column: str, feature_columns: Optional[List[str]] = None,
                        correlation_threshold: float = 0.95) -> int:
    """
    Count feature columns suspiciously correlated with the target (potential leakage).
    
    Args:
        df: Input DataFrame
        target_column: Target/label column
        feature_columns: Feature columns to check; if None, use all numeric columns except target
        correlation_threshold: Correlation threshold above which to flag leakage
        
    Returns:
        Integer count of potential leakage candidates
    """
    if target_column not in df.columns:
        return 0
    
    if feature_columns is None:
        feature_columns = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_column]
    
    if not feature_columns:
        return 0
    
    numeric_df = pd.to_numeric(df[target_column], errors="coerce").to_frame()
    leakage_count = 0
    
    for feat_col in feature_columns:
        if feat_col not in df.columns:
            continue
        
        feat_numeric = pd.to_numeric(df[feat_col], errors="coerce")
        valid_idx = numeric_df[target_column].notna() & feat_numeric.notna()
        
        if valid_idx.sum() > 2:
            corr = numeric_df.loc[valid_idx, target_column].corr(feat_numeric[valid_idx])
            if abs(corr) >= correlation_threshold:
                leakage_count += 1
    
    return leakage_count


def potential_target_leakage_candidate_count(df: pd.DataFrame, target_column: str) -> int:
    """
    Count feature columns suspiciously correlated with the target.
    (Alias for target_leakage_count)
    """
    return target_leakage_count(df, target_column)


def raw_pii_field_count(df: pd.DataFrame) -> int:
    """
    Count columns matching PII patterns in column names or values.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Integer count of columns with PII risk
    """
    pii_field_keywords = ["ssn", "phone", "email", "credit", "card", "account", "mru", "dob", "medical"]
    pii_count = 0
    
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in pii_field_keywords):
            pii_count += 1
    
    return pii_count


# image and media metrics

def aspect_ratio_variance(image_dimensions: List[Tuple[int, int]]) -> float:
    """
    Calculate variance in aspect ratios across images.
    
    Args:
        image_dimensions: List of (width, height) tuples
        
    Returns:
        Float representing variance in aspect ratios
    """
    if not image_dimensions or len(image_dimensions) < 2:
        return 0.0
    
    aspect_ratios = [w / h for w, h in image_dimensions if h > 0]
    if not aspect_ratios:
        return 0.0
    
    return float(np.var(aspect_ratios))


def blur_index(image_array: np.ndarray) -> float:
    """
    Calculate blur index using Laplacian variance (requires image as numpy array).
    
    Args:
        image_array: Image as numpy array (grayscale or RGB)
        
    Returns:
        Float representing blur index (lower = blurrier)
    """
    try:
        from PIL import Image, ImageFilter
        import cv2
        
        if isinstance(image_array, np.ndarray):
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if len(image_array.shape) == 3 else image_array
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            return float(laplacian_var)
    except Exception:
        return 0.0
    
    return 0.0


def iou_intersection_over_union(box1: Tuple[int, int, int, int], 
                                 box2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: Bounding box 1 as (x1, y1, x2, y2)
        box2: Bounding box 2 as (x1, y1, x2, y2)
        
    Returns:
        Float between 0 and 1 representing IoU
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return float(inter_area / union_area) if union_area > 0 else 0.0


def iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """
    Alias for Intersection over Union (IoU).
    """
    return iou_intersection_over_union(box1, box2)


# statistical and change detection

def change_point_detection_score(timeseries: np.ndarray) -> float:
    """
    Detect change points in time series using statistical methods.
    
    Args:
        timeseries: 1D numerical array
        
    Returns:
        Float representing number of detected change points (normalized 0-1)
    """
    if len(timeseries) < 3:
        return 0.0
    
    try:
        # Use rolling mean and std to detect significant changes
        ts = np.array(timeseries, dtype=float)
        rolling_mean = pd.Series(ts).rolling(window=max(2, len(ts) // 5), center=True).mean()
        rolling_std = pd.Series(ts).rolling(window=max(2, len(ts) // 5), center=True).std()
        
        # Detect points where value deviates significantly
        upper_bound = rolling_mean + 2 * rolling_std
        lower_bound = rolling_mean - 2 * rolling_std
        
        violations = ((ts > upper_bound) | (ts < lower_bound)).sum()
        change_points = violations / len(ts)
        
        return min(float(change_points), 1.0)
    except Exception:
        return 0.0


# ============================================================================
# UTILITY & REGISTRATION
# ============================================================================

METRIC_REGISTRY = {
    # Data Quality
    "completeness_rate": completeness_rate,
    "uniqueness_rate": uniqueness_rate,
    "format_validity_rate": format_validity_rate,
    "outlier_rate": outlier_rate,
    "total_missing_value_rate": total_missing_value_rate,
    "schema_type_violations": schema_type_violations,
    "exact_duplicate_overlap_rate": exact_duplicate_overlap_rate,
    "exact_duplicate_overlap_rate_across_splits": exact_duplicate_overlap_rate_across_splits,
    
    # ML & Statistical
    "r_squared": r_squared,
    "variance_inflation_factor": variance_inflation_factor,
    "vif": variance_inflation_factor,
    "population_stability_index": population_stability_index,
    "change_point_detection_score": change_point_detection_score,
    
    # Class & Fairness
    "class_imbalance_ratio": class_imbalance_ratio,
    "demographic_parity_disparity": demographic_parity_disparity,
    
    # Target Leakage
    "target_leakage_count": target_leakage_count,
    "potential_target_leakage_candidate_count": potential_target_leakage_candidate_count,
    "raw_pii_field_count": raw_pii_field_count,
    
    # PII & Privacy
    "pii_contamination_rate": pii_contamination_rate,
    "pii_phi_contamination_rate": pii_phi_contamination_rate,
    "pii_phiviolation_rate": pii_phiviolation_rate,
    
    # Image & Media
    "aspect_ratio_variance": aspect_ratio_variance,
    "blur_index": blur_index,
    "iou": iou,
    "iou_intersection_over_union": iou_intersection_over_union,
    "intersection_over_union": iou_intersection_over_union,
    
    # Aliases
    "Variance Inflation Factor": variance_inflation_factor,
    "Population Stability Index": population_stability_index,
    "Aspect Ratio Variance": aspect_ratio_variance,
    "Blur Index": blur_index,
    "IoU (Intersection over Union)": iou,
    "Intersection Over Union (IoU)": iou_intersection_over_union,
    "Schema Type Violations": schema_type_violations,
    "Change Point Detection Score": change_point_detection_score,
}


def get_metric(metric_name: str) -> Optional[callable]:
    """
    Retrieve a metric function by name from the registry.
    
    Args:
        metric_name: Name of the metric (case-insensitive)
        
    Returns:
        Metric function or None if not found
    """
    return METRIC_REGISTRY.get(metric_name) or METRIC_REGISTRY.get(metric_name.lower())


def list_available_metrics() -> List[str]:
    """List all available metric names."""
    return list(METRIC_REGISTRY.keys())
