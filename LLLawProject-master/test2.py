# Import necessary libraries
from pypdf import PdfReader
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Initialize the tokenizer and model from the pre-trained T5 base model
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Function to read and extract text from a PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text() + '\n\n'  # Ensure separation between pages
    return text

# Function to summarize a section of text
def summarize_section(section_text):
    inputs = tokenizer.encode("summarize: " + section_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4,
                                 early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Main program
if __name__ == "__main__":
    pdf_path = 'C:/Users/sugam/LLLawProject/Previous_case_files/rob_case.pdf'  # Replace with your PDF file path
    full_text = extract_text_from_pdf(pdf_path)

    # Split the text into sections for summarization
    sections = full_text.split('\n\n')  # This is a simple split by double newlines; you may need a more sophisticated approach

    # Summarize each section individually
    summaries = []
    for section in sections:
        if section.strip() != '':
            if len(section) > 512:
                # If the section is too long, split it into chunks and summarize each chunk
                chunks = [section[i:i+512] for i in range(0, len(section), 512)]
                chunk_summaries = [summarize_section(chunk) for chunk in chunks]
                summaries.append('\n\n'.join(chunk_summaries))
            else:
                summaries.append(summarize_section(section))

    # Combine the summaries
    final_summary = '\n\n'.join(summaries)
    print(final_summary)
