import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the DistilBERT model
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Load the CSV file containing text pairs
@st.cache
def load_data():
    df = pd.read_csv("/content/drive/MyDrive/DataNeuron_Text_Similarity.csv")
    return df

# Function to compute similarity score using cosine similarity
def compute_similarity_score(embedding1, embedding2):
    # Compute cosine similarity
    similarity_score = cosine_similarity(embedding1, embedding2)[0][0]
    return similarity_score

# Define the Streamlit app
def main():
    st.title("Text Similarity Scoring")

    # Load the data
    load_data()

    # Input text fields for text1 and text2
    text1 = st.text_area("Enter text 1:", height=100)
    text2 = st.text_area("Enter text 2:", height=100)

    # Compute similarity score when both texts are provided
    if st.button("Calculate Similarity"):
        if text1 and text2:
            # Encode the texts into embeddings
            embedding1 = model.encode(text1)
            embedding2 = model.encode(text2)

            # Compute similarity score
            similarity_score = compute_similarity_score(embedding1.reshape(1, -1), embedding2.reshape(1, -1))

            # Display similarity score
            st.success(f"Similarity score: {similarity_score:.2f}")

            # Add a plot or visualization
            st.bar_chart({"Similarity Score": similarity_score})

# Run the Streamlit app
if __name__ == "__main__":
    main()
