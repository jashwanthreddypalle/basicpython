import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import string

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

def load_text_file(file_path):
    """Load a text file and return its content as a string."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

def preprocess_text(text):
    """Tokenize the text and clean it by removing stopwords and punctuation."""
    # Tokenize text
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    
    return tokens

def display_most_common_words(tokens, n=10):
    """Display the n most common words from the tokens."""
    freq_dist = FreqDist(tokens)
    print(f"Top {n} most common words:")
    for word, freq in freq_dist.most_common(n):
        print(f"{word}: {freq}")

def main():
    file_path = input("Enter the path to the text file: ")
    text = load_text_file(file_path)
    
    if text:
        tokens = preprocess_text(text)
        display_most_common_words(tokens, n=10)

if __name__ == "__main__":
    main()
