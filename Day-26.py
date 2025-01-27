from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(string1, string2):
    # Create a TfidfVectorizer
    vectorizer = TfidfVectorizer()
    
    # Transform the strings into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform([string1, string2])
    
    # Calculate the cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    return similarity[0][0]

# Input strings
string1 = input("Enter the first string: ")
string2 = input("Enter the second string: ")

# Calculate and display the cosine similarity
similarity_score = calculate_cosine_similarity(string1, string2)
print(f"Cosine Similarity: {similarity_score:.4f}")
