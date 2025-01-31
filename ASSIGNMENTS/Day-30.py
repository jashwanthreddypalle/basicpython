from textblob import TextBlob

def perform_sentiment_analysis(text):
    """Perform sentiment analysis on the given text and display the sentiment."""
    # Create a TextBlob object
    blob = TextBlob(text)

    # Get sentiment polarity
    polarity = blob.sentiment.polarity

    # Determine sentiment category
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    print(f"Sentiment: {sentiment} (Polarity: {polarity:.2f})")

def main():
    # Input text from the user
    text = input("Enter the text for sentiment analysis: ")

    # Perform sentiment analysis
    perform_sentiment_analysis(text)

if __name__ == "__main__":
    main()
