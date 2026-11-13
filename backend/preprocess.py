
import pandas as pd

# Load the dataset
df = pd.read_csv("data/Loan_default.csv")

# Remove duplicate records
df = df.drop_duplicates()

# Display dataset information
print("Dataset shape:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nFirst 5 rows:")
print(df.head())
