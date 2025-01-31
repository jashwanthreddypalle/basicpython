import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import string

# Download necessary NLTK data files
nltk.download('punkt')

def load_text_file(file_path):
    """Load a text file and return its content as a string."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

def calculate_term_frequency(text):
    """Tokenize the text and calculate term frequency for each token."""
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove punctuation
    tokens = [word for word in tokens if word.isalnum()]

    # Calculate frequency distribution
    freq_dist = FreqDist(tokens)

    return freq_dist

def display_top_tokens(freq_dist, n=5):
    """Display the top n most frequent tokens."""
    print(f"Top {n} most frequent tokens:")
    for token, freq in freq_dist.most_common(n):
        print(f"{token}: {freq}")

def main():
    file_path = input("Enter the path to the text file: ")
    text = load_text_file(file_path)

    if text:
        freq_dist = calculate_term_frequency(text)
        display_top_tokens(freq_dist, n=5)

if __name__ == "__main__":
    main()