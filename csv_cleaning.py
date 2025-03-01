import pandas as pd
input_file = "fuel-prices-for-be-assessment.csv"
output_file = "fuel-prices-for-be-assessment_updated.csv"
df = pd.read_csv(input_file)
df["combined_address"] = df.apply(lambda row: f"{row['Truckstop Name']}, {row['Address']}, {row['City']}, {row['State']}", axis=1)
df.to_csv(output_file, index=False)
df.to_csv('fuel-prices-for-be-assessment_updated_chunk.csv', index=False)

print(f"Updated CSV saved as {output_file}")

