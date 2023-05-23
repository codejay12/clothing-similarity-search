import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Load the necessary pickle files
tsvt = pickle.load(open("tsvt.pkl", "rb"))
encoding_matrix = pickle.load(open("encoding_matrix.pkl", "rb"))
dictionary = pickle.load(open("dictionary.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# Read the data from the CSV file
data = pd.read_csv("main_data.csv")

# Define the Streamlit app
def main():
    # Create an input element for the user to enter the query
    st.title("Products Recommendation Project")
    query = st.text_input("What are you looking for?")
     # Search button
    if st.button("Search"):
        # Add your search functionality here
        # This code will be executed when the button is clicked
        st.write("Performing search...")

    # Process the query and fetch results when a query is entered
    if query:
        # Preprocess the query text
        query = re.sub('[\s+\d+:\.\)\( ]', ' ', query)
        query = re.sub(r'\S*@\S*\s?', '', query)
        query = query.lower()
        query_tokens = word_tokenize(query)

        # Convert the query into a TF-IDF matrix
        query_mat = tfidf.transform([query])

        # Transform the query TF-IDF matrix using the loaded TruncatedSVD object
        query_lsa = tsvt.transform(query_mat)

        # Calculate cosine similarity between the query and the encoded training data
        similarity_scores = cosine_similarity(query_lsa, encoding_matrix)
        similarity_scores = similarity_scores.reshape(-1)

        # Sort the similarity scores in descending order
        sorted_indices = np.argsort(similarity_scores)[::-1]

        # Get the top-k most similar results (e.g., URLs)
        k = 10
        top_k_results = data.loc[sorted_indices[:k], "url"]

        # Display the top-k results
        st.write("Suggested Products Links ")
        for result in top_k_results:
            st.write(result)

# Run the Streamlit app
if __name__ == "__main__":
    main()
