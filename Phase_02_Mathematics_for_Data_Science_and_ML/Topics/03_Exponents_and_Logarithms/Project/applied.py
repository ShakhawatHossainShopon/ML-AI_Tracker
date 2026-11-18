"""
Phase 02: Mathematics for Data Science and ML
Topic 03: Exponents and Logarithms — Practical ML Lab
File: applied.py (সহজ ও বাস্তবমুখী রেফারেন্স কোড)

🎯 এই প্রজেক্টের উদ্দেশ্য:
মেশিন লার্নিং-এ সূচক (Exponent) এবং লগারিদম (Logarithm) কেন ব্যবহার হয় তা বোঝা:
১. ডেটার বড় আউটলায়ার বা বৈষম্য কমাতে Log-Transformation (np.log1p)
২. এক্সপোনেন্ট দিয়ে প্রেডিকশন ফিরিয়ে আনা (np.expm1)
৩. এক্সপোনেন্ট ব্যবহার করে Softmax ক্লাসিফিকেশন প্রবাবিলিটি তৈরি
৪. PyTorch-এর সাথে শতভাগ মিল যাচাই
"""

import numpy as np
import torch


print("=" * 65)
print("🔢 মেশিন লার্নিং সূচক ও লগারিদম ইঞ্জিন (Exponents & Logarithms)")
print("=" * 65)


# ==============================================================================
# ধাপ ০১: বাস্তব স্কিউড ডেটায় Log-Transformation প্রয়োগ
# ==============================================================================
# বাস্তব জীবনে আয়ের ডেটাতে বেশিরভাগ সাধারণ সংখ্যা থাকে, কিন্তু কিছু সংখ্যা কোটি টাকার হয়
salaries = np.array([25000.0, 32000.0, 48000.0, 1200000.0])

# log1p মানে log(1 + x) — এটি ০ বা খুব ছোট সংখ্যার ক্ষেত্রেও নিরাপদ
log_transformed_salaries = np.log1p(salaries)

# expm1 মানে e^x - 1 — এটি লগকে আবার আসল ডলারে রূপান্তর করে
recovered_salaries = np.expm1(log_transformed_salaries)

print(f"আসল আয়ের ডেটা (Outlier সহ) : {salaries}")
print(f"লগ রূপান্তরের পর (সহজ স্কেল) : {np.round(log_transformed_salaries, 2)}")
print(f"এক্সপোনেন্ট দিয়ে আসল ডেটা ফেরত: {recovered_salaries.astype(int)}")
print("💡 টিপস: লগ নিলে বিশাল সংখ্যাগুলো ছোট হয়ে যায়, ফলে মডেল সহজে ট্রেন হয়।\n")


# ==============================================================================
# ধাপ ০২: এক্সপোনেন্ট দিয়ে Softmax ফাংশন (মাল্টি-ক্লাস ক্লাসিফিকেশন)
# ==============================================================================
# ৩টি ক্লাসের জন্য নিউরাল নেটওয়ার্কের র' স্কোর (Logits)
# ক্লাসসমূহ: [বিড়াল, কুকুর, পাখি]
logits = np.array([2.0, 1.0, 0.1])

# ১. প্রতিটি মানকে e-এর পাওয়ার বানানো (Negative সংখ্যাও Positive হয়ে যায়)
exp_scores = np.exp(logits)

# ২. যোগফল দিয়ে ভাগ করে শতকরা সম্ভাবনায় (Probability) রূপান্তর
softmax_probabilities = exp_scores / np.sum(exp_scores)

print("🎯 Softmax ক্লাসিফিকেশন ফলাফল:")
print(f"   র' লগিট স্কোর [বিড়াল, কুকুর, পাখি] : {logits}")
print(f"   e^z মান (Exponential)             : {np.round(exp_scores, 4)}")
print(f"   Softmax সম্ভাবনা (Probabilities)   : {np.round(softmax_probabilities * 100, 2)} %")
print(f"   সব সম্ভাবনার যোগফল                : {np.sum(softmax_probabilities):.1f} (১০০%)\n")


# ==============================================================================
# ধাপ ০৩: PyTorch ফ্রেমওয়ার্ক দিয়ে ভেরিফিকেশন
# ==============================================================================
# PyTorch-এর torch.softmax দিয়ে একই গণনা করা
logits_tensor = torch.tensor(logits)
torch_softmax = torch.softmax(logits_tensor, dim=0).numpy()

print("--- PyTorch ফ্রেমওয়ার্ক ভেরিফিকেশন ---")
print(f"PyTorch Softmax মান : {np.round(torch_softmax * 100, 2)} %")

# মিল পরীক্ষা
if np.allclose(softmax_probabilities, torch_softmax):
    print("✅ ফলাফল: আমাদের গাণিতিক Softmax এবং PyTorch-এর গণনা ১০০% হুবহু এক!\n")
    print("🎉 অভিনন্দন! আপনি সফলভাবে সূচক ও লগারিদমের বাস্তব ML ব্যবহার বুঝে ফেলেছেন।")
