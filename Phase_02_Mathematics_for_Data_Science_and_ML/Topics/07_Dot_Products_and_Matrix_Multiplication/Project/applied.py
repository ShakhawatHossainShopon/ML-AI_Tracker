"""
Phase 02: Mathematics for Data Science and ML
Topic 07: Dot Products & Matrix Multiplication
File: applied.py (সম্পূর্ণ কার্যকর রেফারেন্স সল্যুশন)

🎯 উদ্দেশ্য:
১. ভেক্টর ডট প্রোডাক্ট: ১ জন কাস্টমারের কেনাকাটার মোট বিল বের করা।
২. ম্যাট্রিক্স-ভেক্টর গুণ: একসাথে ৩ জন কাস্টমারের আলাদা আলাদা বিল এক ক্লিকে বের করা।
৩. ম্যাট্রিক্স-ম্যাট্রিক্স গুণ (@): দুটি ভিন্ন স্টোরে কাস্টমারদের খরচের তুলনা করা।
"""

import numpy as np


print("=" * 65)
print("🛒 Smart Store Billing & Revenue Engine (Dot & Matrix Mult)")
print("=" * 65)


# ==============================================================================
# ধাপ ০১: ভেক্টর ডট প্রোডাক্ট (Vector Dot Product: 1D @ 1D)
# ১ জন কাস্টমারের কার্ট: [৩ কেজি আপেল, ২ লিটার দুধ, ৫টি ডিম]
# একক পণ্যের মূল্য ভেক্টর: [৫০ টাকা, ৯০ টাকা, ১২ টাকা]
# ==============================================================================
cart = np.array([3, 2, 5])
unit_prices = np.array([50, 90, 12])

# Dot product: (3*50) + (2*90) + (5*12) = 150 + 180 + 60 = 390
single_customer_bill = np.dot(cart, unit_prices)

print("--- Step 1: Single Customer Cart (Dot Product) ---")
print("Items Bought (Apples, Milk, Eggs) :", cart)
print("Item Unit Prices (Taka)           :", unit_prices)
print("Total Bill for Customer 1         :", single_customer_bill)
print()


# ==============================================================================
# ধাপ ০২: ম্যাট্রিক্স ও ভেক্টর গুণ (Matrix-Vector Multiplication: 2D @ 1D)
# ৩ জন কাস্টমারের পুরো টেবিল (Shape: 3 x 3)
# Row 0: Customer 1 [3, 2, 5]
# Row 1: Customer 2 [1, 4, 10]
# Row 2: Customer 3 [0, 2, 12]
# ==============================================================================
customers_orders = np.array([
    [3, 2, 5],
    [1, 4, 10],
    [0, 2, 12]
])

# প্রতিটি সারির সাথে প্রাইস ভেক্টরের ডট প্রোডাক্ট
all_customers_bills = np.dot(customers_orders, unit_prices)

print("--- Step 2: Multiple Customers (Matrix-Vector Mult) ---")
print("Customers Orders Table (Rows: Customers, Cols: Items):\n", customers_orders)
print("\nGenerated Bills for [Customer 1, Customer 2, Customer 3]:")
print(all_customers_bills)
print()


# ==============================================================================
# ধাপ ০৩: ম্যাট্রিক্স ও ম্যাট্রিক্স গুণ (Matrix-Matrix Mult using @ operator: 2D @ 2D)
# দুটি ভিন্ন স্টোরের দামের টেবিল (Shape: 3 x 2)
# Col 0: Store A (ধানমন্ডি ব্রাঞ্চ) -> [50, 90, 12]
# Col 1: Store B (উত্তরা ব্রাঞ্চ)   -> [48, 95, 10]
# ==============================================================================
store_prices = np.array([
    [50, 48],
    [90, 95],
    [12, 10]
])

# পাইথনের আধুনিক @ অপারেটর দিয়ে ম্যাট্রিক্স গুণ
comparison_table = customers_orders @ store_prices

print("--- Step 3: Multi-Store Comparison (Matrix @ Matrix) ---")
print("Store Prices Table (Row: Item, Col 0: Store A, Col 1: Store B):\n", store_prices)
print("\nComparison Table (Rows: Customers, Col 0: Store A Bill, Col 1: Store B Bill):")
print(comparison_table)
print("\n✅ Dot products and matrix multiplications completed successfully!\n")
