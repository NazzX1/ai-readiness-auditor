"""
Structured metrics computation module for AI Readiness Auditor.

Provides 35+ metrics for data quality, ML evaluation, fairness, privacy, and governance assessments.
"""

from src.structured_metrics.metrics_calculator import (
    # Data Quality Metrics
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
    
    # Target Leakage & PII Field Counting
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
    iou,
    iou_intersection_over_union,
    
    # Utilities
    METRIC_REGISTRY,
    get_metric,
    list_available_metrics,
)

__all__ = [
    # Data Quality
    "completeness_rate",
    "uniqueness_rate",
    "format_validity_rate",
    "outlier_rate",
    "total_missing_value_rate",
    "schema_type_violations",
    "exact_duplicate_overlap_rate",
    "exact_duplicate_overlap_rate_across_splits",
    
    # ML & Statistical
    "r_squared",
    "variance_inflation_factor",
    "population_stability_index",
    "change_point_detection_score",
    
    # Class & Fairness
    "class_imbalance_ratio",
    "demographic_parity_disparity",
    
    # Target Leakage & PII
    "target_leakage_count",
    "potential_target_leakage_candidate_count",
    "raw_pii_field_count",
    
    # PII & Privacy
    "pii_contamination_rate",
    "pii_phi_contamination_rate",
    "pii_phiviolation_rate",
    
    # Image & Media
    "aspect_ratio_variance",
    "blur_index",
    "iou",
    "iou_intersection_over_union",
    
    # Utilities
    "METRIC_REGISTRY",
    "get_metric",
    "list_available_metrics",
]
