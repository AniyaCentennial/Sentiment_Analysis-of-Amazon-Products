import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Define positive and negative keywords for rule-based prediction
positive_keywords = {"good", "great", "excellent", "amazing", "awesome", "love", "fantastic", "happy"}
negative_keywords = {"bad", "worst", "poor", "terrible", "awful", "hate", "disappointing", "not good"}

# Preprocessing function to clean text
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text).lower()
    words = text.split()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Load the saved model and vectorizer
model_path = "D:/Projects_DataAnalyst/Sentiment_Analysis/model/random_forest_model.pkl"
vectorizer_path = "D:/Projects_DataAnalyst/Sentiment_Analysis/model/tfidf_vectorizer.pkl"

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Test the model with new input
def predict_sentiment(review):
    # Rule-based prediction for short reviews
    tokens = set(review.lower().split())
    if len(tokens) <= 3:  # Handle short reviews
        if tokens & positive_keywords:
            return "Positive"
        elif tokens & negative_keywords:
            return "Negative"

    # Preprocess and vectorize the input review
    review = preprocess_text(review)
    review_vector = vectorizer.transform([review])
    prediction = model.predict(review_vector)[0]
    
    # Map prediction to sentiment
    sentiment_map = {1: "Positive", 0: "Neutral", -1: "Negative"}
    return sentiment_map.get(prediction, "Unknown")

# Input review for testing
while True:
    user_input = input("\nEnter a review (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("Exiting...")
        break
    sentiment = predict_sentiment(user_input)
    print(f"Predicted Sentiment: {sentiment}")
