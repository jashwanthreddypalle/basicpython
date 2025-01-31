import spacy

def perform_ner(text):
    """Perform Named Entity Recognition (NER) on the given text and print entities and their types."""
    # Load the SpaCy English model
    nlp = spacy.load("en_core_web_sm")

    # Process the text
    doc = nlp(text)

    # Extract and print entities and their types
    print("Named Entities and their Types:")
    for ent in doc.ents:
        print(f"{ent.text}: {ent.label_}")

def main():
    # Input text from the user
    text = input("Enter the text for NER analysis: ")

    # Perform Named Entity Recognition
    perform_ner(text)

if __name__ == "__main__":
    main()
