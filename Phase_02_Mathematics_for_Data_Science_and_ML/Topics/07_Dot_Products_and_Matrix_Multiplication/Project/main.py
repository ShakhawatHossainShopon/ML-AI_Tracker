import numpy as np

cart = np.array([3, 2, 5])
unit_prices = np.array([50, 90, 12])

single_customer_bill = np.dot(cart, unit_prices)

print("--- Step 1: Single Customer Cart (Dot Product) ---")
print("Items Bought (Apples, Milk, Eggs) :", cart)
print("Item Unit Prices (Taka)           :", unit_prices)
print("Total Bill for Customer 1         :", single_customer_bill)
