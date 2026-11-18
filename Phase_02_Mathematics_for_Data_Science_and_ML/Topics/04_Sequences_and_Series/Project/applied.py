"""
Phase 02: Mathematics for Data Science and ML
Topic 04: Sequences and Series — Simple Practical ML Lab
File: applied.py (একদম সহজ প্রোগ্রামিং সমাধান)

🎯 সহজ উদ্দেশ্য:
ম্যাথের কোনো জটিল সূত্র ছাড়া, সাধারণ পাইথন লুপ দিয়ে অনুক্রম ও ধারা বোঝা:
১. Sequence (অনুক্রম): প্রতি রাউন্ডে ভুল অর্ধেক কমার একটি সাধারণ লিস্ট [100, 50, 25, 12, 6]
২. Series (ধারা): এই লিস্টের সবগুলোর মোট যোগফল (sum)
৩. Moving Average (গড় ধারা): দুটি সংখ্যার গড় বের করে ওঠানামা মসৃণ করা
"""

# ==============================================================================
# ধাপ ০১: অনুক্রম (Sequence) — সাধারণ পাইথন লিস্ট
# ==============================================================================
current_error = 100
error_history = []

for epoch in range(5):
    error_history.append(current_error)
    # প্রতি রাউন্ডে ভুল অর্ধেক (৫০%) হয়ে যাচ্ছে
    current_error = int(current_error * 0.5)

print("--- Step 1: Tracking Error as a Sequence ---")
print("Error History per Epoch:", error_history)
print()


# ==============================================================================
# ধাপ ০২: ধারা (Series) — লিস্টের সব সংখ্যার যোগফল
# ==============================================================================
# অনুক্রমের সংখ্যাগুলো যোগ করলেই তাকে ধারা বলে
total_accumulated_error = sum(error_history)

print("--- Step 2: Summing Sequence to make a Series ---")
print("Total Accumulated Error:", total_accumulated_error)
print()


# ==============================================================================
# ধাপ ০৩: মুভিং এভারেজ (Moving Average)
# ==============================================================================
# পাশাপাশি দুটি পয়েন্টের গড় বের করে ডেটা স্মুথ করা
moving_avg = []
for i in range(1, len(error_history)):
    avg = (error_history[i - 1] + error_history[i]) / 2.0
    moving_avg.append(avg)

print("--- Step 3: Simple Moving Average Series ---")
print("Smoothed Values:", moving_avg)
