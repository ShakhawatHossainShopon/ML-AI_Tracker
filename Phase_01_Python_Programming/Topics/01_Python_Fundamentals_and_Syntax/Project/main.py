"""
Phase 01: Python Programming
Topic 01: Python Fundamentals and Syntax - Practical Lab
Project: AI Training Batch Stream Validator & Memory Profiler

Instructions:
Complete the functions marked with TODO below according to the visual dashboard (index.html).
Verify your solution by running:
    python test_project.py
"""

import sys
import json
from typing import Dict, Any, Tuple, List

# ==============================================================================
# Task 1: Bitwise Flag Constants
# ==============================================================================
# TODO: Define the 4 bitwise constants as specified:
FLAG_VALID = 1 << 0          # Decimal 1
FLAG_CONTAINS_NAN = 1 << 1   # Decimal 2
FLAG_DIM_MISMATCH = 1 << 2   # Decimal 4
FLAG_ZERO_WEIGHT = 1 << 3    # Decimal 8


# ==============================================================================
# Task 2: Validate AI Training Sample
# ==============================================================================
def validate_sample(raw_sample: Dict[str, Any], expected_dim: int = 4) -> Tuple[Dict[str, Any], int]:
    """
    Validates, casts and computes bitwise status flags for a single training sample.

    Args:
        raw_sample: Raw dictionary with keys: 'sample_id', 'features', 'label', 'loss_weight'
        expected_dim: Expected number of features in the feature vector

    Returns:
        Tuple of (cleaned_sample, status_code)
    """
    # TODO 1: Clean and uppercase sample_id (as str)
    # TODO 2: Cast label to int
    # TODO 3: Cast loss_weight to float rounded to 4 decimal places
    # TODO 4: Inspect features list:
    #         - Check for None, "nan", or non-finite values -> FLAG_CONTAINS_NAN
    #         - Check len(features) != expected_dim -> FLAG_DIM_MISMATCH
    #         - If numbers are valid, cast each to float rounded to 4 decimals
    # TODO 5: If loss_weight <= 0.0 -> FLAG_ZERO_WEIGHT
    # TODO 6: If no error flags triggered -> add FLAG_VALID

    cleaned_sample = {}
    status_code = 0
    return cleaned_sample, status_code


# ==============================================================================
# Task 3: Profile Object Memory
# ==============================================================================
def profile_object_memory(obj: Any, label: str) -> Dict[str, Any]:
    """
    Inspects object memory address, reference count, and mutability classification.

    Args:
        obj: Any Python object
        label: Descriptive label

    Returns:
        Dict with 'label', 'type', 'memory_id_hex', 'ref_count', 'is_mutable'
    """
    # TODO:
    # 1. Obtain hex representation of id(obj) -> hex(id(obj))
    # 2. Get type name -> type(obj).__name__
    # 3. Adjusted reference count -> sys.getrefcount(obj) - 1
    # 4. Check if type is mutable (list, dict, set, bytearray) -> bool
    
    return {
        "label": label,
        "type": type(obj).__name__,
        "memory_id_hex": hex(id(obj)),
        "ref_count": 0,
        "is_mutable": False
    }


# ==============================================================================
# Task 4: Format Batch Telemetry Summary Table
# ==============================================================================
def format_batch_summary(cleaned_samples: List[Dict[str, Any]], status_codes: List[int]) -> str:
    """
    Formats an aligned fixed-width ASCII telemetry table using modern Python f-strings.

    Columns:
    | Sample ID   | Dim | Label | Loss Weight | Status Flags |
    """
    # TODO: Construct formatted string with headers, separators, and data rows
    return ""


# ==============================================================================
# Main Demonstration Runner
# ==============================================================================
if __name__ == "__main__":
    print("=== AI Training Batch Stream Validator & Memory Profiler ===")
    try:
        with open("sample_batches.json", "r", encoding="utf-8") as f:
            raw_batches = json.load(f)
    except FileNotFoundError:
        print("Error: sample_batches.json not found.")
        sys.exit(1)

    cleaned_list = []
    status_list = []

    for raw in raw_batches:
        cleaned, status = validate_sample(raw, expected_dim=4)
        cleaned_list.append(cleaned)
        status_list.append(status)

    print("\nProcessed Training Samples:")
    print(format_batch_summary(cleaned_list, status_list))

    print("\nMemory Profile of First Sample Object:")
    if cleaned_list:
        print(profile_object_memory(cleaned_list[0], "First Training Sample"))
