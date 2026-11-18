import numpy as np

warehouse_1 = np.array([
    [100, 150, 200],
    [300, 350, 400]
])

warehouse_2 = np.array([
    [20, 30, 25],
    [40, 50, 60]
])

total_inventory = warehouse_1 + warehouse_2

print("--- Step 1: Matrix Addition ---")
print("Warehouse 1:")
print(warehouse_1)
print("\nWarehouse 2:")
print(warehouse_2)
print("\nTotal Inventory (Warehouse 1 + Warehouse 2):")
print(total_inventory)
print("Table Shape:", total_inventory.shape)

tax_multiplier = 1.15
total_with_tax = total_inventory * tax_multiplier

print("\n--- Step 2: Scalar Multiplication (15% Tax) ---")
print("Tax Multiplier:", tax_multiplier)
print("Inventory Cost After 15% Tax:")
print(np.round(total_with_tax, 2))

transposed_inventory = total_with_tax.T

print("\n--- Step 3: Matrix Transpose ---")
print("Original Table Shape (Items x Outlets)   :", total_with_tax.shape)
print("Transposed Table Shape (Outlets x Items) :", transposed_inventory.shape)
print("Transposed Inventory Table:")
print(np.round(transposed_inventory, 2))
