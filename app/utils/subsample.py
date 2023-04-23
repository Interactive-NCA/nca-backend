import pandas as pd

df = pd.read_csv("~/Downloads/models_exp11_trained_archive.csv")

# Sort the DataFrame based on "objective" column in descending order
df_sorted = df.sort_values("objective", ascending=False)

# Select the first 100 rows of the sorted DataFrame
df_top100 = df_sorted.head(100).copy()

# Save the new DataFrame to a CSV file in a folder
df_top100.to_csv("~/Downloads/trained_archive.csv", index=False)