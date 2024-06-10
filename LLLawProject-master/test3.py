import os
import PyPDF2
from sentence_transformers import SentenceTransformer, util

def find_relevant_cases(file, previous_cases_folder, model_name="all-mpnet-base-v2"):
    """Compares a query case file with previous cases and recommends relevant ones."""

    # Get the filename of the uploaded file
    query_file = file.filename

    # Get all the PDF files in the previous cases folder
    previous_case_files = [os.path.join(previous_cases_folder, f) for f in os.listdir(previous_cases_folder) if
                           f.endswith('.pdf') and f != query_file]

    # Load PDF files and extract text
    query_text = extract_text_from_pdf(file)
    previous_case_texts = [
        extract_text_from_pdf(os.path.join(previous_cases_folder, file)) for file in previous_case_files
    ]

    # Load semantic embedding model
    embedder = SentenceTransformer(model_name)

    # Generate embeddings for text
    query_embedding = embedder.encode([query_text], convert_to_tensor=True)
    previous_case_embeddings = embedder.encode(previous_case_texts, convert_to_tensor=True)

    # Calculate pairwise similarities
    similarities = util.pytorch_cos_sim(query_embedding, previous_case_embeddings)

    # Filter and rank relevant cases
    relevant_cases = []
    for i, similarity_score in enumerate(similarities[0]):
        if similarity_score.item() > 0.15 and similarity_score.item() < 0.9999:  # Adjust threshold as needed
            previous_case_file = previous_case_files[i]
            previous_case_text = previous_case_texts[i]
            # Round the similarity score to 4 decimal places
            rounded_score = round(float(similarity_score.item()), 4)
            relevant_cases.append((rounded_score, previous_case_file, previous_case_text))


    return sorted(relevant_cases, reverse=True)


def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""

    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text
