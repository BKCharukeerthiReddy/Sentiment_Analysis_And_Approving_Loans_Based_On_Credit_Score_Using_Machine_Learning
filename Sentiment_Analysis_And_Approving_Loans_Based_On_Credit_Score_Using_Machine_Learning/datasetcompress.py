import pandas as pd

# Load your original CSV file (replace 'yourfile.csv' with your actual file name)
df = pd.read_csv('Reviews.csv')

# Get the first 450000 rows
df_subset = df.head(450000)

# Save to a new CSV file
df_subset.to_csv('first_450000_rows.csv', index=False)

print("✅ First 450,000 rows saved to 'first_450000_rows.csv'")

