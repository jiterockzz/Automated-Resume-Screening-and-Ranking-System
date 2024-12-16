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
from keras.models import load_model
import openai


app = Flask(__name__, static_url_path='/static')


# Load model and vectorizer
model = joblib.load('models/model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

def extract_text_from_pdf(file):
    file_stream = io.BytesIO(file.read())
    return extract_text(file_stream)

def extract_text_from_docx(file):
    return docx2txt.process(file)

def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W+', ' ', text)
    text = text.lower()
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def compute_similarity(text1, text2):
    vectorized_texts = vectorizer.transform([text1, text2])
    return cosine_similarity(vectorized_texts[0:1], vectorized_texts[1:2])[0][0]

def rank_resumes_from_csv(csv_file, job_description, limit):
    df = pd.read_csv(csv_file)
    
    # Ensure the original index is preserved
    df.reset_index(inplace=True)  # This creates a new column 'index' that holds the original index
    
    # Rename the column 'index' to 'Original Index'
    df.rename(columns={'index': 'Original Index'}, inplace=True)
    
    # Clean the resume text
    df['Resume'] = df['Resume'].apply(clean_text)
    
    # Vectorize the job description and resumes
    vectorizer = TfidfVectorizer()
    job_description_vec = vectorizer.fit_transform([job_description])
    resume_vec = vectorizer.transform(df['Resume'])
    
    # Compute similarities
    similarities = cosine_similarity(job_description_vec, resume_vec).flatten()
    df['Similarity'] = similarities
    
    # Rank resumes based on similarity
    ranked_df = df.sort_values(by='Similarity', ascending=False).reset_index(drop=True)
    
    return ranked_df

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load DialoGPT model
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# @app.route('/chat', methods=['GET', 'POST'])
# def chat():
#     if request.method == 'POST':
#         user_message = request.form.get('message')
        
#         # Encode input and generate response
#         inputs = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors="pt")
#         outputs = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
#         response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
        
#         return render_template('chat.html', user_message=user_message, bot_response=response)
#     return render_template('chat.html')
@app.route('/chat', methods=['GET', 'POST'])
def ask_chatbot(question, context):
    # Example code to interact with the GPT model (like DialoGPT)
    input_text = f"Context: {context}\nUser Question: {question}"
    inputs = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")
    outputs = chat_model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response



@app.route('/ranking', methods=['GET','POST'])
def ranking_page():
    if request.method == "POST":
        csv_file = request.files.get('csv_file')
        job_description = request.form.get('job_description')
        limit = int(request.form.get('limit', 10))  # Default to 10 if not provided
        if not csv_file or not job_description:
            return render_template('ranking.html', error='Please provide all required inputs.')
        
        try:
            ranked_resumes = rank_resumes_from_csv(csv_file, job_description, limit)
            return render_template('ranking.html', ranked_resumes=ranked_resumes.to_dict(orient='records'))
        except Exception as e:
            return render_template('ranking.html', error=str(e))
    return render_template('ranking.html')

@app.route('/', methods=['GET', 'POST'])
def home():
    chatbot_response = None  # To hold the chatbot's response
    context = ""  # Context from the resume and job description
    message = None  # To hold the classification result
    similarity = None  # To hold the similarity score

    if request.method == 'POST':
        if 'resume' in request.files:
            file = request.files['resume']
            if file.filename == '':
                return render_template('index.html', error='No file selected')

            # Extract text from resume based on file type
            if file.filename.endswith('.pdf'):
                resume_text = extract_text_from_pdf(file)
            elif file.filename.endswith('.docx'):
                resume_text = extract_text_from_docx(file)
            else:
                return render_template('index.html', error='Unsupported file type')

            # Get the job description from the form
            job_description = request.form.get('job_description', '')
            if not job_description:
                return render_template('index.html', error='Job description is required.')

            # Compute similarity between the resume and the job description
            similarity = compute_similarity(resume_text, job_description)
            cleaned_text = clean_text(resume_text)

            # Vectorize the cleaned resume text for classification
            vectorized_text = vectorizer.transform([cleaned_text])
            prediction = model.predict(vectorized_text)

            # Prepare the context for chatbot
            context = (
                f"Resume Content: {resume_text[:1000]}... "  # Limit the length for clarity
                f"Job Description: {job_description}. "
                f"Similarity Score: {similarity:.2f}. Prediction: {prediction[0]}."
            )

            # Set the classification result and similarity score
            message = f'Resume classified as: {prediction[0]}'
            similarity = f'Job description suitability: {similarity:.2f}'

        # Handle chatbot queries (if any)
        question = request.form.get('chatbot_question', '')
        if question and context:
            print(f"Question: {question}")
            print(f"Context: {context}")
            chatbot_response = ask_chatbot(question, context)
            print(f"Chatbot Response: {chatbot_response}")


    return render_template(
        'index.html',
        message=message,
        similarity=similarity,
        chatbot_response=chatbot_response,
        context=context
    )

@app.route('/suitability', methods=['GET', 'POST'])
def check_suitability():
    chatbot_response = None  # To hold the chatbot's response
    context = ""  # Context from the resume and job description

    if request.method == 'POST':
        # Handle file upload for resume analysis
        if 'resume' in request.files:
            file = request.files['resume']
            if file.filename == '':
                return render_template('suitability.html', error='No file selected')

            if file.filename.endswith('.pdf'):
                resume_text = extract_text_from_pdf(file)
            elif file.filename.endswith('.docx'):
                resume_text = extract_text_from_docx(file)
            else:
                return render_template('suitability.html', error='Unsupported file type')

            job_description = request.form.get('job_description', '')
            similarity = compute_similarity(resume_text, job_description)
            cleaned_text = clean_text(resume_text)
            vectorized_text = vectorizer.transform([cleaned_text])
            prediction = model.predict(vectorized_text)

            # Prepare context for chatbot
            context = (
                f"Resume Content: {resume_text[:1000]}... "  # Limit the length for clarity
                f"Job Description: {job_description}. "
                f"Similarity Score: {similarity:.2f}. Prediction: {prediction[0]}."
            )

            return render_template(
                'suitability.html',
                message=f'Resume classified as: {prediction[0]}',
                similarity=f'Job description suitability: {similarity:.2f}',
                context=context
            )

        # Handle chatbot queries
        question = request.form.get('chatbot_question', '')
        if question and context:
            chatbot_response = ask_chatbot(question, context)

    return render_template('suitability.html', chatbot_response=chatbot_response)


if __name__ == '__main__':
    app.run(debug=True)
