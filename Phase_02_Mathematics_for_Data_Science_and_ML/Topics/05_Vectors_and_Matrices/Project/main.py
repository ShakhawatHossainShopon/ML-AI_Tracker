import numpy as np

single_user = np.array([28, 50, 720])

users_table = np.array([
    [28, 50, 720],
    [35, 80, 680],
    [42, 120, 750]
])

print("--- Step 1: Vector vs Matrix Creation ---")
print("Single User Vector (1D):", single_user)
print("Vector Shape          :", single_user.shape)
print("Users Matrix Table (2D):")
print(users_table)
print("Matrix Shape (Rows, Cols):", users_table.shape)

first_user = users_table[0, :]
all_salaries = users_table[:, 1]

print("\n--- Step 2: Slicing Rows and Columns ---")
print("First User Data (Row 0) :", first_user)
print("All Users Salary (Col 1):", all_salaries)

column_averages = np.mean(users_table, axis=0)

print("\n--- Step 3: Column Averages ---")
print("Average [Age, Salary, Score]:", column_averages)
