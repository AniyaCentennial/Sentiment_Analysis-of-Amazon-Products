from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the dataset
csv_file_path = "D:/Projects_DataAnalyst/Sentiment_Analysis/data/cleaned_reviews.csv"
data = pd.read_csv(csv_file_path)

# Clean the 'sentiments' column
data['sentiments'] = data['sentiments'].str.strip().str.title()

# Drop rows with missing data in required columns
data = data.dropna(subset=['sentiments', 'cleaned_review'])
data['cleaned_review'] = data['cleaned_review'].astype(str)

# Check sentiment distribution
print("Sentiment Distribution:")
print(data['sentiments'].value_counts())

# Bar Chart: Sentiment Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='sentiments', data=data, order=["Positive", "Neutral", "Negative"], palette="viridis")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.savefig('D:/Projects_DataAnalyst/Sentiment_Analysis/outputs/visualization/sentiment_distribution_bar.png')
plt.show()

# Pie Chart: Sentiment Distribution
plt.figure(figsize=(6, 6))
data['sentiments'].value_counts().plot.pie(
    autopct='%1.1f%%', colors=['#1f77b4', '#ff7f0e', '#d62728'], explode=(0.1, 0, 0)
)
plt.title("Sentiment Distribution (Pie Chart)")
plt.ylabel("")
plt.savefig('D:/Projects_DataAnalyst/Sentiment_Analysis/outputs/visualization/sentiment_distribution_pie.png')
plt.show()

# Generate Word Clouds for each sentiment
for sentiment in ['Positive', 'Neutral', 'Negative']:
    # Combine reviews for the current sentiment
    sentiment_text = " ".join(
        data[data['sentiments'] == sentiment]['cleaned_review']
    )
    
    if sentiment_text.strip():  # Check if there is data for this sentiment
        wordcloud = WordCloud(background_color='white', max_words=200).generate(sentiment_text)
        
        plt.figure(figsize=(8, 6))
        plt.title(f"Word Cloud for {sentiment} Sentiment")
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.savefig(f'D:/Projects_DataAnalyst/Sentiment_Analysis/outputs/visualization/wordcloud_{sentiment.lower()}.png')
        plt.show()
    else:
        print(f"No data available for {sentiment} sentiment. Skipping word cloud generation.")
