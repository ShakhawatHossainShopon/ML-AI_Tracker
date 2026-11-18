"""
Phase 02: Mathematics for Data Science and ML
Topic 05: Vectors and Matrices — Simple Practical Lab
File: applied.py (সম্পূর্ণ কার্যকর রেফারেন্স সল্যুশন)

🎯 সহজ উদ্দেশ্য:
একজন প্রোগ্রামারের চোখে ভেক্টর ও ম্যাট্রিক্স কী তা বোঝা:
১. Vector (১D): একজন ইউজারের বৈশিষ্ট্যের লিস্ট [বয়স, বেতন, ক্রেডিট স্কোর]
২. Matrix (২D): একাধিক ইউজারের ডেটা টেবিল (Rows = ইউজার, Columns = ফিচার্স)
৩. স্লাইসিং: টেবিল থেকে নির্দিষ্ট ইউজার বা কলাম আলাদা করা
৪. কলামের গড়: সব ইউজারের গড় বয়স, বেতন ও স্কোর হিসাব করা
"""

import numpy as np


print("=" * 65)
print("📊 প্রোগ্রামারের চোখে ভেক্টর ও ম্যাট্রিক্স (Vectors & Matrices)")
print("=" * 65)


# ==============================================================================
# ধাপ ০১: ভেক্টর (1D) এবং ম্যাট্রিক্স (2D) তৈরি
# ==============================================================================
# ১টি ভেক্টর = ১ জন ইউজারের ৩টি ফিচার: [বয়স, মাসিক আয় (হাজারে), ক্রেডিট স্কোর]
single_user = np.array([28, 50, 720])

# ১টি ম্যাট্রিক্স = ৩ জন ইউজারের পুরো ডেটা টেবিল (Row = ইউজার, Col = ফিচার)
users_table = np.array([
    [28, 50, 720],   # User 1
    [35, 80, 680],   # User 2
    [42, 120, 750]   # User 3
])

print("--- Step 1: Vector vs Matrix Creation ---")
print("Single User Vector (1D):", single_user)
print("Vector Shape          :", single_user.shape)
print("\nUsers Matrix Table (2D):\n", users_table)
print("Matrix Shape (Rows, Cols):", users_table.shape)
print()


# ==============================================================================
# ধাপ ০২: ম্যাট্রিক্স স্লাইসিং (নির্দিষ্ট ইউজার বা ফিচার বের করা)
# ==============================================================================
# প্রথম ইউজার (১ম সারি)
first_user = users_table[0, :]

# সব ইউজারের মাসিক আয় (২য় কলাম)
all_salaries = users_table[:, 1]

print("--- Step 2: Slicing Rows and Columns ---")
print("First User Data (Row 0) :", first_user)
print("All Users Salary (Col 1):", all_salaries)
print()


# ==============================================================================
# ধাপ ০৩: কলামগুলোর গড় (Average) বের করা
# ==============================================================================
# সব ইউজারের গড় বয়স, গড় বেতন ও গড় স্কোর বের করা
column_averages = np.mean(users_table, axis=0)

print("--- Step 3: Column Averages ---")
print("Average [Age, Salary, Score]:", column_averages)
print("✅ ভেক্টর ও ম্যাট্রিক্স হলো ডেটা সাজিয়ে রাখার সাধারণ টেবিল!\n")
