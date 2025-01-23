import pandas as pd

# Load the dataset
csv_file_path = "D:/Projects_DataAnalyst/Sentiment_Analysis/data/sentiment_analysis_results.csv"
data = pd.read_csv(csv_file_path)

# Check columns
print(data.columns)

# Save the cleaned dataset (if needed)
tableau_ready_path = "D:/Projects_DataAnalyst/Sentiment_Analysis/data/tableau_ready.csv"
data.to_csv(tableau_ready_path, index=False)
print(f"Dataset prepared for Tableau: {tableau_ready_path}")
