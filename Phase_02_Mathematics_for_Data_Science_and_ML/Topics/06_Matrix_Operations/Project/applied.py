"""
Phase 02: Mathematics for Data Science and ML
Topic 06: Matrix Operations (ম্যাট্রিক্স অপারেশন)
File: applied.py (সম্পূর্ণ কার্যকর রেফারেন্স সল্যুশন)

🎯 উদ্দেশ্য:
১. ম্যাট্রিক্স যোগ (Matrix Addition): দুটি গুদামের পণ্যের খরচ টেবিল একই পজিশনে যোগ করা।
২. স্কেলার গুণ (Scalar Multiplication): পুরো টেবিলের সব ডেটায় এক লাইনে ১৫% ট্যাক্স যোগ করা।
৩. ট্রান্সপোজ (Transpose): টেবিলের সারি (Rows) এবং কলাম (Columns) অদলবদল করা।
"""

import numpy as np


print("=" * 65)
print("📦 Warehouse Inventory & Tax Manager (Matrix Operations)")
print("=" * 65)


# ==============================================================================
# ধাপ ০১: ম্যাট্রিক্স যোগ (Element-wise Addition)
# দুটি গুদামের ডেটা: ২ প্রকার পণ্য x ৩টি আউটলেট (Shape: 2 x 3)
# ==============================================================================
warehouse_1 = np.array([
    [100, 150, 200],
    [300, 350, 400]
])

warehouse_2 = np.array([
    [20, 30, 25],
    [40, 50, 60]
])

# দুটি একই আকারের টেবিল সরাসরি যোগ করা
total_inventory = warehouse_1 + warehouse_2

print("--- Step 1: Matrix Addition (Warehouse 1 + Warehouse 2) ---")
print("Warehouse 1 Base Cost:\n", warehouse_1)
print("\nWarehouse 2 Additional Cost:\n", warehouse_2)
print("\nTotal Combined Inventory Cost (Element-wise Sum):")
print(total_inventory)
print("Inventory Shape:", total_inventory.shape)
print()


# ==============================================================================
# ধাপ ০২: স্কেলার গুণ (Scalar Multiplication / Tax Calculation)
# পুরো টেবিলের প্রতিটি সেলে ১৫% ট্যাক্স যোগ করতে ১.১৫ দিয়ে গুণ করা
# ==============================================================================
tax_multiplier = 1.15
total_with_tax = total_inventory * tax_multiplier

print("--- Step 2: Scalar Multiplication (Applying 15% Tax) ---")
print("Tax Multiplier:", tax_multiplier)
print("Inventory Cost After 15% Tax:")
print(np.round(total_with_tax, 2))
print()


# ==============================================================================
# ধাপ ০৩: ম্যাট্রিক্স ট্রান্সপোজ (Matrix Transpose: .T)
# সারিকে কলাম এবং কলামকে সারিতে উল্টে ফেলা (Shape পরিবর্তিত হয়ে 3 x 2 হবে)
# ==============================================================================
transposed_inventory = total_with_tax.T

print("--- Step 3: Matrix Transpose (Swapping Rows and Columns) ---")
print("Original Table Shape (Items x Outlets)   :", total_with_tax.shape)
print("Transposed Table Shape (Outlets x Items) :", transposed_inventory.shape)
print("\nTransposed Inventory Table (Outlets as Rows):")
print(np.round(transposed_inventory, 2))
print("\n✅ Matrix operations completed successfully!\n")
