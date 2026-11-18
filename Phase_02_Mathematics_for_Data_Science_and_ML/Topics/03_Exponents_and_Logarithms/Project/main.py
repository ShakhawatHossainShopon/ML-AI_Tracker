import numpy as np

salaries = np.array([25000.0, 32000.0, 48000.0, 1200000.0])

log_salaries = np.log1p(salaries)
recovered_salaries = np.expm1(log_salaries)

print("--- Step 1: Log-Transformation & Inverse Exponent ---")
print("Original Salaries:", salaries)
print("Log-Transformed  :", np.round(log_salaries, 2))
print("Recovered Values :", recovered_salaries.astype(int))
