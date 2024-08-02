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

app = Flask(__name__)

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
    if request.method == 'POST':
        if 'resume' in request.files:
            file = request.files['resume']
            if file.filename == '':
                return render_template('index.html', error='No file selected')
            
            if file.filename.endswith('.pdf'):
                resume_text = extract_text_from_pdf(file)
            elif file.filename.endswith('.docx'):
                resume_text = extract_text_from_docx(file)
            else:
                return render_template('index.html', error='Unsupported file type')

            job_description = request.form.get('job_description', '')
            similarity = compute_similarity(resume_text, job_description)
            cleaned_text = clean_text(resume_text)
            vectorized_text = vectorizer.transform([cleaned_text])
            prediction = model.predict(vectorized_text)

            return render_template('index.html', 
                                   message=f'Resume classified as: {prediction[0]}',
                                   similarity=f'Job description suitability: {similarity:.2f}')
    return render_template('index.html')

@app.route('/suitability', methods=['GET', 'POST'])
def check_suitability():
    if request.method == 'POST':
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

            return render_template('suitability.html', 
                                   message=f'Resume classified as: {prediction[0]}',
                                   similarity=f'{similarity:.2f}')
    return render_template('suitability.html')

if __name__ == '__main__':
    app.run(debug=True)
