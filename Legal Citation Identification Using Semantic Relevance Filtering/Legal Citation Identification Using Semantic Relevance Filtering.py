# -*- coding: utf-8 -*-
"""IGNICULT GAMELABS Assignment.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1OztnSTME54jkPXVwh-kDVVpDEk1mwBMc
"""

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load English language model
nlp = spacy.load("en_core_web_sm")

def semantic_relevance_filtering(text, keywords, threshold=0.02):
    relevant_citations = []
    relevant_sentences = []

    # Tokenize and preprocess the input text
    doc = nlp(text)

    # Extract named entities from the input text
    entities = [ent.text for ent in doc.ents]
    print("Named Entities:", entities)

    # Combine keywords and named entities for TF-IDF calculation
    corpus = [text] + entities + keywords
    print("Corpus:", corpus)

    # Calculate TF-IDF vectors for the input text and entities+keywords
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    print("TF-IDF Matrix Shape:", tfidf_matrix.shape)

    # Calculate cosine similarity between input text and entities+keywords
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    print("Similarities:", similarities)

    # Identify relevant entities+keywords based on cosine similarity threshold
    relevant_entities_keywords = [corpus[i + 1] for i, similarity in enumerate(similarities[0]) if similarity >= threshold]
    print("Relevant Entities+Keywords:", relevant_entities_keywords)

    # Extract citations from relevant entities+keywords
    for entity_keyword in relevant_entities_keywords:
        citations = extract_citations(entity_keyword)  # Your citation extraction logic here
        relevant_citations.extend(citations)

    return relevant_citations

# Replace this function with your actual citation extraction logic
def extract_citations(entity_keyword):
    # Your citation extraction logic here
    # Example: Extract citations using regular expressions
    import re
    pattern = r'[A-Z][a-z]+ v\. [A-Z][a-z]+'
    citations = re.findall(pattern, entity_keyword)
    return citations

# Example usage
text = "The Church of Hope seeks to maintain its tax-exempt status. The IRS argues the Church engages in substantial commercial activities."
keywords = ["religious", "tax-exempt", "IRS", "commercial activity"]

relevant_citations = semantic_relevance_filtering(text, keywords)
print("Relevant Citations:", relevant_citations)

"""Here's a summary of the above code:

1. **Load Language Model:** The code begins by loading the English language model provided by spaCy.

2. **Semantic Relevance Filtering Function:** The `semantic_relevance_filtering` function is defined, which takes the input text, a list of keywords, and an optional threshold parameter as input. This function aims to identify relevant citations in the text based on semantic relevance to the keywords.

3. **Tokenization and Named Entity Extraction:** The input text is tokenized and processed using the spaCy language model. Named entities (such as organizations, locations, etc.) are extracted from the text.

4. **TF-IDF Calculation:** The function combines the input text, named entities, and keywords to form a corpus. TF-IDF (Term Frequency-Inverse Document Frequency) vectors are then calculated for the corpus using scikit-learn's `TfidfVectorizer`.

5. **Cosine Similarity Calculation:** Cosine similarity is calculated between the TF-IDF vector of the input text and the TF-IDF vectors of named entities and keywords. This measures the similarity of the input text with each named entity and keyword.

6. **Relevant Entities and Keywords:** Entities and keywords with cosine similarity scores above the threshold are considered relevant. These relevant entities and keywords are then used to extract potential citations.

7. **Citation Extraction:** The `extract_citations` function is called for each relevant entity or keyword to extract potential citations. In this example, the `extract_citations` function uses regular expressions to find patterns that match citation formats.

8. **Example Usage:** An example text about a legal case involving a church and the IRS is provided. The keywords related to the case are also provided. The `semantic_relevance_filtering` function is called with this example text and keywords.

9. **Output:** The function outputs the relevant citations found in the input text based on semantic relevance to the provided keywords.

This code demonstrates a basic approach to identifying relevant citations in a legal text based on semantic relevance to specified keywords. However, the effectiveness of this approach may vary depending on factors such as the quality of the language model, the choice of keywords, and the complexity of the text.
"""

import pdfplumber

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.

    Args:
    - pdf_path (str): Path to the PDF file.

    Returns:
    - text (str): Extracted text from the PDF.
    """
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Example usage
pdf_path = "/content/drive/MyDrive/dataset_assignment.pdf"
text = extract_text_from_pdf(pdf_path)

# Replace the example keywords with relevant keywords from your dataset
keywords = ["interfere","demonetisation", "liabilities" ]

# Call your Semantic Relevance Filtering function with the extracted text and keywords
relevant_citations = semantic_relevance_filtering(text, keywords)

print("\nRelevant Citations:", relevant_citations)

"""Here's a summary of the above code:

1. **PDF Text Extraction Function:** The code defines a function named extract_text_from_pdf to extract text from a PDF file. It uses the pdfplumber library to open the PDF file and iterates through each page, extracting text from each page and concatenating it to form the complete text of the document.

2. **Example Usage:** The function is called with the path to the PDF file (pdf_path) to extract text from the PDF.

3. **Keyword Definition:** Relevant keywords related to the document are defined in a list named keywords. These keywords are crucial for identifying relevant citations within the extracted text.

4. **Semantic Relevance Filtering:** The semantic_relevance_filtering function (not shown in the provided code snippet) is called with the extracted text and the defined keywords. This function aims to identify relevant citations in the text based on semantic relevance to the provided keywords.

5. **Output:** The function outputs the relevant citations found in the extracted text based on semantic relevance to the provided keywords. These citations are then printed.

The code allows for easy extraction of text from PDF documents and identification of relevant citations based on specified keywords. It provides a convenient way to analyze legal documents and extract relevant information for further processing or analysis.
"""

from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate_rag_system(dataset, keywords, threshold=0.02):
    predicted_citations = []
    actual_citations = []

    for document in dataset:
        text = document['text']
        annotated_citations = document['citations']

        # Apply semantic relevance filtering algorithm
        relevant_citations = semantic_relevance_filtering(text, keywords, threshold)
        predicted_citations.append(relevant_citations)
        actual_citations.append(annotated_citations)

    # Flatten the lists of citations
    predicted_citations_flat = [citation for sublist in predicted_citations for citation in sublist]
    actual_citations_flat = [citation for sublist in actual_citations for citation in sublist]

    # Calculate precision, recall, and F1 score
    precision = precision_score(actual_citations_flat, predicted_citations_flat, average='micro')
    recall = recall_score(actual_citations_flat, predicted_citations_flat, average='micro')
    f1 = f1_score(actual_citations_flat, predicted_citations_flat, average='micro')

    return precision, recall, f1

# Example dataset (replace with your actual dataset)
dataset = [
    {'text': text , 'citations': ['India v. Reserve', 'Cellular v. Union', 'Garg v. Union', 'Another v. Union', 'Bommai v. Union', 'Nath v. State', 'Aggarwal v. Reserve', 'Shah v. Reserve', 'India v. State', 'Scindia v. Union', 'India v. Reserve', 'Cellular v. Union', 'Shah v. Reserve', 'Gandhi v. Union', 'Mathew v. South', 'Jain v. Union', 'Krishnan v. State', 'Another v. Union', 'Garg v. Union', 'Another v. Union', 'India v. Sankalchand', 'Quareshi v. State', 'Shah v. The']},
]

# Define relevant keywords
keywords = ["interfere","demonetisation", "liabilities" ]

# Evaluate the RAG system
precision, recall, f1 = evaluate_rag_system(dataset, keywords)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

"""The code evaluates the performance of a Relevance-Adapted Generation (RAG) system for identifying relevant citations from legal documents using precision, recall, and F1 score metrics. It defines an `evaluate_rag_system` function that iterates over a dataset, applies a semantic relevance filtering algorithm to extract relevant citations based on specified keywords, and compares the predicted citations with actual annotated citations. The function then calculates precision, recall, and F1 score, providing a quantitative assessment of the RAG system's effectiveness in citation identification."""
