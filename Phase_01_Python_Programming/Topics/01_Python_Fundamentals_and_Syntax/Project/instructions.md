# AI Lab: Training Batch Stream Validator & Memory Profiler

**Topic 01:** Python Fundamentals and Syntax  
**Phase 01:** Python Programming  
**Type:** Hands-On Implementation Lab  

---

## 🎯 Project Overview
In machine learning training pipelines, bad batches containing NaNs, wrong feature dimensions, or invalid weights cause silent failures or loss divergence during multi-GPU distributed training.

In this project, you will implement a pure-Python training batch validator and memory profiler:
1. **Define Bitwise Status Flags:** Pack sample anomaly conditions into an integer bitmask.
2. **Implement `validate_sample`:** Clean sample IDs, validate feature dimensions, detect NaNs, and check loss weights.
3. **Implement `profile_object_memory`:** Profile object address (`hex(id())`), reference counts, and mutability.
4. **Implement `format_batch_summary`:** Generate fixed-width telemetry tables using Python f-strings.

---

## 🧪 Verification
Run the automated test runner:
```bash
python test_project.py
```
Run the live batch processing script:
```bash
python main.py
```
When all assertions pass, open the master [index.html](../../../../index.html) and mark Topic 01 as 🟢 **Done**!
