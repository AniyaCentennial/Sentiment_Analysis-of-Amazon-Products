import pandas as pd
from textblob import TextBlob

# Load the dataset
csv_file_path = 'D:/Projects_DataAnalyst/Sentiment_Analysis/data/cleaned_reviews.csv'
data = pd.read_csv(csv_file_path)

# Ensure the 'cleaned_review' column exists
if 'cleaned_review' not in data.columns:
    raise ValueError("The dataset must contain a 'cleaned_review' column.")

# Handle missing or non-string values
data['cleaned_review'] = data['cleaned_review'].fillna('')  # Replace NaN with empty strings
data['cleaned_review'] = data['cleaned_review'].astype(str)  # Convert all entries to strings

# Function to classify sentiment based on polarity
def get_sentiment(text):
    try:
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0:
            return "Positive"
        elif polarity == 0:
            return "Neutral"
        else:
            return "Negative"
    except Exception as e:
        return "Unknown"  # Fallback for unexpected issues

# Apply the function to the 'cleaned_review' column
data['sentiments'] = data['cleaned_review'].apply(get_sentiment)

# Display sentiment counts
print("\nSentiment Distribution:")
print(data['sentiments'].value_counts())

# Save the results
output_file_path = 'D:/Projects_DataAnalyst/Sentiment_Analysis/data/sentiment_analysis_results.csv'
data.to_csv(output_file_path, index=False)
print(f"\nSentiment analysis results saved to: {output_file_path}")
