!pip install sentence-transformers
!pip install streamlit

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the DistilBERT model
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Function to compute similarity scores for a batch of text pairs
@st.cache_data
def compute_similarity_scores(text_pairs):
    similarity_scores = []
    for text_pair in text_pairs:
        embedding1 = model.encode(text_pair[0])
        embedding2 = model.encode(text_pair[1])
        similarity_score = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]
        similarity_scores.append(similarity_score)
    return similarity_scores

# Define the Streamlit app
def main():
    st.title("Text Similarity Scoring")
    st.markdown("---")

    st.markdown(
        """
        <style>
        .main-container {
            background-color: #f0f2f6;
            padding: 2rem;
        }
        .stButton>button {
            background-color: #2ecc71;
            color: white;
            font-size: 1rem;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #27ae60;
        }
        .stTextInput>div>div>input {
            background-color: #ffffff;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-size: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Ask user if they want to upload a CSV file or enter two independent texts
    option = st.radio("Choose an option:", ("Upload CSV file", "Enter two texts"))

    if option == "Upload CSV file":
        st.markdown("Please upload a CSV file with 'text1' and 'text2' columns.")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            text_pairs = df[['text1', 'text2']].values.tolist()
            batch_size = 100  # Adjust batch size based on performance
            similarity_scores = []
            with st.spinner("Calculating similarity scores..."):
                for i in range(0, len(text_pairs), batch_size):
                    batch_text_pairs = text_pairs[i:i+batch_size]
                    similarity_scores.extend(compute_similarity_scores(batch_text_pairs))
            df['similarity_score'] = similarity_scores
            st.dataframe(df)
    else:
        st.markdown("Enter two texts to check their similarity:")
        text1 = st.text_area("Enter text 1:", height=100)
        text2 = st.text_area("Enter text 2:", height=100)
        if st.button("Calculate Similarity"):
            if text1 and text2:
                similarity_score = compute_similarity_scores([(text1, text2)])[0]
                st.success(f"Similarity score: {similarity_score:.2f}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
