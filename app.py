from flask import Flask, request, render_template
from werkzeug.exceptions import RequestEntityTooLarge
from PyPDF2 import PdfReader
import re
import pickle
import spacy
import pandas as pd

app = Flask(__name__)

# Limit upload size to 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# Load categorization model
rf_classifier_categorization = pickle.load(open('models/rf_classifier_categorization.pkl', 'rb'))
tfidf_vectorizer_categorization = pickle.load(open('models/tfidf_vectorizer_categorization.pkl', 'rb'))

# Load spaCy model
nlp = spacy.load('en_core_web_sm')


# Clean resume function
def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText


# Category prediction
def predict_category(resume_text):
    text = cleanResume(resume_text)
    tfidf = tfidf_vectorizer_categorization.transform([text])
    return rf_classifier_categorization.predict(tfidf)[0]


# PDF to text utility
def pdf_to_text(file):
    reader = PdfReader(file)
    return ''.join(page.extract_text() or '' for page in reader.pages)


# Contact extraction
def extract_contact_number_from_resume(text):
    m = re.search(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", text)
    return m.group() if m else None


def extract_email_from_resume(text):
    m = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
    return m.group() if m else None


# Extract name from resume
def extract_name_from_resume(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            return ent.text
    return None


# Extract skills from resume
def extract_skills_from_resume(text):
    skills = ['Python', 'Java', 'JavaScript', 'C++', 'SQL', 'HTML', 'CSS', 'Machine Learning', 'Data Analysis']
    text = cleanResume(text).lower()
    found_skills = [skill for skill in skills if skill.lower() in text]
    return found_skills


# Extract education from resume
def extract_education_from_resume(text):
    degrees = ['Bachelors', 'Masters', 'PhD', 'Bachelor of Science', 'Master of Science']
    text = cleanResume(text).lower()
    found_degrees = [degree for degree in degrees if degree.lower() in text]
    return found_degrees


@app.errorhandler(RequestEntityTooLarge)
def handle_overlarge(e):
    return render_template("resume.html", message="File too large (max 16 MB)."), 413


@app.route('/')
def resume():
    return render_template("resume.html")


@app.route('/pred', methods=['POST'])
def pred():
    if 'resume' not in request.files:
        return render_template("resume.html", message="No resume file uploaded.")
    file = request.files['resume']
    if file.filename == '':
        return render_template("resume.html", message="No file selected.")

    filename = file.filename.lower()
    try:
        if filename.endswith('.pdf'):
            text = pdf_to_text(file)
        elif filename.endswith('.txt'):
            text = file.read().decode('utf-8')
        else:
            return render_template("resume.html", message="Invalid format. Upload PDF or TXT.")
    except Exception as e:
        return render_template("resume.html", message=f"Error processing file: {e}")

    if not text.strip():
        return render_template("resume.html", message="No extractable text found in file.")

    predicted_category = predict_category(text)
    phone = extract_contact_number_from_resume(text)
    email = extract_email_from_resume(text)
    skills = extract_skills_from_resume(text)
    education = extract_education_from_resume(text)
    name = extract_name_from_resume(text)

    return render_template(
        'resume.html',
        predicted_category=predicted_category,
        phone=phone,
        email=email,
        name=name,
        extracted_skills=skills,
        extracted_education=education
    )


if __name__ == '__main__':
    app.run(debug=True)
