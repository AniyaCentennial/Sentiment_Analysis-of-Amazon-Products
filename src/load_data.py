import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
csv_file_path = "D:/Projects_DataAnalyst/Sentiment_Analysis/data/cleaned_reviews.csv"

# Load the dataset
try:
    data = pd.read_csv(csv_file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    raise FileNotFoundError(f"Dataset not found at path: {csv_file_path}")

# Inspect the dataset
print("First 5 rows of the dataset:")
print(data.head())
print("\nDataset info:")
print(data.info())

# Ensure the necessary column exists
if 'cleaned_review' not in data.columns:
    raise ValueError("The dataset must contain a 'cleaned_review' column.")