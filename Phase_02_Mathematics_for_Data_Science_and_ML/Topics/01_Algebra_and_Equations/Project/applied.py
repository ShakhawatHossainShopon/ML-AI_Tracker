"""
Phase 02: Mathematics for Data Science and ML
Topic 01: Algebra and Equations — Practical ML Lab
File: applied.py (বাস্তব সেলস প্রেডিকশন মডেল — সম্পূর্ণ শর্ট ও কমপ্লিট সল্যুশন)

🎯 এই ফাইলের উদ্দেশ্য:
টপিকের সবগুলো কনসেপ্ট (SymPy, NumPy, Normal Equation, Scikit-learn) খুব সংক্ষেপে
বাস্তব বিজ্ঞাপনী সেলস প্রেডিকশন মডেলের মাধ্যমে দেখানো হয়েছে।
"""

import json
import numpy as np
import sympy as sp
from sklearn.linear_model import LinearRegression


print("=" * 65)
print("📊 ML প্রজেক্ট: বিজ্ঞাপন বাজেট থেকে সেলস প্রেডিকশন (Short & Complete)")
print("=" * 65)


# ==============================================================================
# ধাপ ০১: SymPy দিয়ে মডেলের সমীকরণ ডিফাইন করা (Symbolic Touch)
# ==============================================================================
# ML মডেলের মূল সমীকরণকে প্রতীকীভাবে ডিফাইন করা:
tv_sym, soc_sym, w0_sym, w1_sym, w2_sym = sp.symbols('TV Social w0 w1 w2')
sales_equation = sp.Eq(w0_sym + w1_sym * tv_sym + w2_sym * soc_sym, sp.symbols('Sales'))

print(f"১. মডেল সমীকরণ (SymPy): {sales_equation}\n")


# ==============================================================================
# ধাপ ০২: ডেটাসেট লোড ও ডিজাইন ম্যাট্রিক্স (X) তৈরি
# ==============================================================================
with open("advertising_data.json", "r", encoding="utf-8") as f:
    campaigns = json.load(f)

tv = np.array([row["tv_budget"] for row in campaigns], dtype=float)
social = np.array([row["social_budget"] for row in campaigns], dtype=float)
sales = np.array([row["sales"] for row in campaigns], dtype=float)

# প্রথম কলামে বায়াস (সব 1) রেখে X ম্যাট্রিক্স এবং y ভেক্টর সাজানো
X = np.column_stack([np.ones_like(tv), tv, social])
y = sales

print(f"২. মোট ডেটা পয়েন্ট: {len(y)}, ডিজাইন ম্যাট্রিক্স X শেপ: {X.shape}\n")


# ==============================================================================
# ধাপ ০৩: NumPy দিয়ে Normal Equation সমাধান (বিনা লুপে ট্রেনিং)
# ==============================================================================
# সূত্র: w = (X^T * X)^(-1) * X^T * y
XTX = X.T @ X
XTy = X.T @ y
w = np.linalg.solve(XTX, XTy)

base_sales, tv_weight, social_weight = w[0], w[1], w[2]

print("৩. Normal Equation থেকে প্রাপ্ত অপটিমাল ওয়েটস:")
print(f"   - বেইস সেলস (w0) : {base_sales:.4f}k")
print(f"   - টিভি প্রভাব (w1): {tv_weight:.4f}")
print(f"   - সোশ্যাল প্রভাব (w2): {social_weight:.4f}\n")


# ==============================================================================
# ধাপ ০৪: নতুন ক্যাম্পেইনে প্রেডিকশন
# ==============================================================================
# নতুন বাজেট: টিভি ১৫০k, সোশ্যাল ৪০k
new_tv, new_social = 150.0, 40.0
predicted_sales = base_sales + (tv_weight * new_tv) + (social_weight * new_social)

print(f"৪. নতুন ক্যাম্পেইনে প্রেডিক্টেড সেলস: ${predicted_sales:.2f}k ডলার\n")


# ==============================================================================
# ধাপ ০৫: Scikit-learn ফ্রেমওয়ার্ক দিয়ে ভেরিফিকেশন
# ==============================================================================
sklearn_model = LinearRegression().fit(np.column_stack([tv, social]), y)
sklearn_pred = sklearn_model.predict([[new_tv, new_social]])[0]

print(f"৫. Scikit-learn ভেরিফিকেশন: Intercept={sklearn_model.intercept_:.4f}, Coefs={np.round(sklearn_model.coef_, 4)}")
print(f"   Scikit-learn প্রেডিকশন : ${sklearn_pred:.2f}k ডলার")

if abs(predicted_sales - sklearn_pred) < 1e-5:
    print("   🎉 নিখুঁত মিল! গণিতের সমীকরণ ও Scikit-learn ১০০% এক!\n")
