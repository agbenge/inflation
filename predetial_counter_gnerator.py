import pandas as pd
import os

# Create data folder if it doesn't exist
data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

# Election cycle parameters
start_year = 1999
end_year = 2026
cycle_length_years = 4

# Create empty list to store data
data = []

# Generate months from June 1999 to May 2026
current_year = start_year
current_month = 6  # June

# Counter starts at 1
cycle_counter = 1

while True:
    # Add current year and month with counter to data
    data.append({
        "Year": current_year,
        "Month": current_month,
        "Counter": cycle_counter
    })
    
    # Move to next month
    current_month += 1
    if current_month > 12:
        current_month = 1
        current_year += 1
    
    # Stop at May 2026
    if current_year == 2026 and current_month == 6:
        break
    
    # Update cycle counter at every 4-year cycle starting June
    if (current_year - start_year) % cycle_length_years == 0 and current_month == 6:
        cycle_counter += 1

# Convert list to DataFrame
df = pd.DataFrame(data)

# Reverse order: recent to old
df = df.iloc[::-1].reset_index(drop=True)

# Save to Excel
output_file = os.path.join(data_folder, "nigeria_presidential_election_cycles.xlsx")
df.to_excel(output_file, index=False)

print(f"Election cycle dataset saved to {output_file} (recent to old order)")