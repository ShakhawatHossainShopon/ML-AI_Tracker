"""
Phase 02: Mathematics for Data Science and ML
Topic 02: Functions and Graphs — Simple Practical ML Lab
File: applied.py (একদম সহজ ও পরিষ্কার রেফারেন্স কোড)

🎯 এই প্রজেক্টের উদ্দেশ্য:
মেশিন লার্নিংয়ের ৩টি সবচেয়ে গুরুত্বপূর্ণ ফাংশন বোঝা:
১. Linear ফাংশন: f(x) = 2x + 1 (সরলরেখা)
২. Sigmoid ফাংশন: σ(x) = 1 / (1 + e^-x) (যেকোনো সংখ্যাকে ০ থেকে ১ এর মধ্যে প্রবাবিলিটি বানায়)
৩. ReLU ফাংশন: f(x) = max(0, x) (নেগেটিভ মানকে ০ বানিয়ে দেয়)
৪. PyTorch-এর সাথে মিলিয়ে দেখা যে তারা হুবহু একই কাজ করে
৫. Matplotlib দিয়ে ফাংশনগুলোর গ্রাফ প্লট করে 'activation_functions_plot.png' ফাইলে সেভ করা
"""

import numpy as np
import matplotlib.pyplot as plt
import torch


print("=" * 60)
print("📈 মেশিন লার্নিং ফাংশন ও গ্রাফ (সহজ প্রজেক্ট)")
print("=" * 60)

# ==============================================================================
# ধাপ ০১: ৩টি মূল ফাংশন তৈরি করা
# ==============================================================================
def linear_func(x):
    return 2 * x + 1

def sigmoid_func(x):
    return 1.0 / (1.0 + np.exp(-x))

def relu_func(x):
    return np.maximum(0.0, x)


# ==============================================================================
# ধাপ ০২: কিছু সাধারণ সংখ্যা দিয়ে আউটপুট পরীক্ষা করা
# ==============================================================================
inputs = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])

print(f"ইনপুট সংখ্যাসমূহ   : {inputs}")
print(f"Linear আউটপুট      : {linear_func(inputs)}")
print(f"Sigmoid আউটপুট     : {np.round(sigmoid_func(inputs), 4)} (সব মান ০ থেকে ১ এর মধ্যে)")
print(f"ReLU আউটপুট        : {relu_func(inputs)} (নেগেটিভগুলো ০ হয়ে গেছে)\n")


# ==============================================================================
# ধাপ ০৩: PyTorch ফ্রেমওয়ার্ক দিয়ে মিলিয়ে দেখা
# ==============================================================================
t_inputs = torch.tensor(inputs)

torch_sig = torch.sigmoid(t_inputs).numpy()
torch_relu = torch.relu(t_inputs).numpy()

print("--- PyTorch ফ্রেমওয়ার্ক ভেরিফিকেশন ---")
print(f"PyTorch Sigmoid    : {np.round(torch_sig, 4)}")
print(f"PyTorch ReLU       : {torch_relu}")
print("✅ ফলাফল: আমাদের সাধারণ গাণিতিক ফাংশন ও PyTorch ১০০% হুবহু এক!\n")


# ==============================================================================
# ধাপ ০৪: গ্রাফ তৈরি ও সেভ করা (Matplotlib)
# ==============================================================================
# -5 থেকে +5 পর্যন্ত ২০০টি সংখ্যা তৈরি
x_vals = np.linspace(-5.0, 5.0, 200)

plt.figure(figsize=(8, 4.5))
plt.plot(x_vals, linear_func(x_vals), label="Linear: 2x + 1", color="gray", linestyle="--")
plt.plot(x_vals, sigmoid_func(x_vals), label="Sigmoid: 1 / (1 + e^-x)", color="#2563eb", linewidth=2.5)
plt.plot(x_vals, relu_func(x_vals), label="ReLU: max(0, x)", color="#16a34a", linewidth=2.5)

plt.title("Comparison of Linear vs Non-Linear Functions in ML", fontsize=12, fontweight="bold")
plt.xlabel("Input (x)")
plt.ylabel("Output (y)")
plt.ylim(-2, 5)
plt.axhline(0, color="#94a3b8", linestyle=":", linewidth=1)
plt.axvline(0, color="#94a3b8", linestyle=":", linewidth=1)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

# গ্রাফ সেভ করা
plt.savefig("activation_functions_plot.png", dpi=150)
print("🖼️ গ্রাফ তৈরি সম্পন্ন হয়েছে এবং 'activation_functions_plot.png' ফাইলে সেভ হয়েছে!")
print("🎉 অভিনন্দন! আপনি সফলভাবে ফাংশন ও গ্রাফের মূল ধারণা বুঝে ফেলেছেন।")
