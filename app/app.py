from flask import Flask, request, render_template
import joblib
import io
import pandas as pd
from pdfminer.high_level import extract_text
import docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelForCausalLM, BartTokenizer, BartForConditionalGeneration
import torch

app = Flask(__name__, static_url_path='/static')

# Load model and vectorizer
model = joblib.load('models/model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# Load chatbot model (DialoGPT) using PyTorch
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Load BART for Summarization
summarization_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
summarization_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Function to extract text from PDF
def extract_text_from_pdf(file):
    file_stream = io.BytesIO(file.read())
    return extract_text(file_stream)

# Function to extract text from DOCX
def extract_text_from_docx(file):
    return docx2txt.process(file)

# Function to clean text
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W+', ' ', text)
    text = text.lower()
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Function to compute similarity
def compute_similarity(text1, text2):
    vectorized_texts = vectorizer.transform([text1, text2])
    return cosine_similarity(vectorized_texts[0:1], vectorized_texts[1:2])[0][0]

# Function to summarize text using BART model
def summarize_text(text):
    inputs = summarization_tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = summarization_model.generate(inputs, max_length=200, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Chatbot logic using PyTorch
def ask_chatbot(question, context):
    try:
        if not context:
            return "No context provided. Please analyze the resume first."

        input_text = f"Context: {context}\nUser Question: {question}"

        inputs = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")
        attention_mask = torch.ones(inputs.shape, dtype=torch.long)

        # Generate response with max_new_tokens instead of max_length
        outputs = chat_model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=50,  # Generate up to 50 new tokens
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )

        response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
        return response
    except Exception as e:
        print(f"Error in ask_chatbot: {e}")
        return "Sorry, I couldn't generate a response."

@app.route('/ranking', methods=['GET', 'POST'])
def ranking_page():
    if request.method == "POST":
        csv_file = request.files.get('csv_file')
        job_description = request.form.get('job_description')
        if not csv_file or not job_description:
            return render_template('ranking.html', error='Please provide all required inputs.')

        try:
            df = pd.read_csv(csv_file)
            df['Resume'] = df['Resume'].apply(clean_text)
            job_vec = vectorizer.transform([job_description])
            resumes_vec = vectorizer.transform(df['Resume'])
            similarities = cosine_similarity(job_vec, resumes_vec).flatten()
            df['Similarity'] = similarities
            ranked_df = df.sort_values(by='Similarity', ascending=False).reset_index(drop=True)
            return render_template('ranking.html', ranked_resumes=ranked_df.to_dict(orient='records'))
        except Exception as e:
            return render_template('ranking.html', error=str(e))
    return render_template('ranking.html')

@app.route('/', methods=['GET', 'POST'])
def home():
    chatbot_response = None  # Holds the chatbot's response
    context = None  # Stores resume and job description data
    message = None  # Resume classification result
    similarity = None  # Resume-job description similarity score

    # Check if the form is submitted
    if request.method == 'POST':
        # Handle Resume Analysis First
        if 'resume' in request.files:  # Resume uploaded
            file = request.files['resume']
            if file.filename == '':
                return render_template('index.html', error='No file selected')

            # Extract text from uploaded file
            if file.filename.endswith('.pdf'):
                resume_text = extract_text_from_pdf(file)
            elif file.filename.endswith('.docx'):
                resume_text = extract_text_from_docx(file)
            else:
                return render_template('index.html', error='Unsupported file type')

            # Get Job Description
            job_description = request.form.get('job_description', '')
            if not job_description:
                return render_template('index.html', error='Job description is required.')

            # Clean and Process the Resume
            similarity = compute_similarity(resume_text, job_description)
            cleaned_text = clean_text(resume_text)
            vectorized_text = vectorizer.transform([cleaned_text])
            prediction = model.predict(vectorized_text)

            # Summarize the resume text
            summarized_text = summarize_text(resume_text)

            # Save the context for chatbot
            context = (
                f"Resume Content: {summarized_text[:1000]}... "  # Shorten resume for clarity
                f"Job Description: {job_description}. "
                f"Similarity Score: {similarity:.2f}. Prediction: {prediction[0]}."
            )

            # Set classification result and similarity score
            message = f'Resume classified as: {prediction[0]}'
            similarity = f'Job description suitability: {similarity:.2f}'

            # Pass context to the form
            return render_template(
                'index.html',
                message=message,
                similarity=similarity,
                context=context
            )

        # Handle Chatbot Query After Resume Analysis
        question = request.form.get('chatbot_question', '')  # Chatbot input
        context = request.form.get('context', '')  # Retrieve existing context
        if question and context:
            chatbot_response = ask_chatbot(question, context)
        else:
            chatbot_response = "Please analyze a resume first before asking questions."

    return render_template(
        'index.html',
        message=message,
        similarity=similarity,
        chatbot_response=chatbot_response,
        context=context
    )

# Test the chatbot function
test_response = ask_chatbot("What is AI?", "Artificial Intelligence is about creating smart systems.")
print(f"Test Response: {test_response}")


if __name__ == '__main__':
    app.run(debug=True)
