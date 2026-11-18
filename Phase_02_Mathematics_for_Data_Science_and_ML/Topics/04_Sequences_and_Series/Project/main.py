current_error = 100
error_history = []

for epoch in range(5):
    error_history.append(current_error)
    current_error = int(current_error * 0.5)

print("--- Step 1: Tracking Error as a Sequence ---")
print("Error History per Epoch:", error_history)
