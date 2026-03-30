# Install libraries
!pip install PyPDF2 scikit-learn

# Import
from google.colab import files
import PyPDF2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Upload resume file
uploaded = files.upload()

# Read PDF file
resume_text = ""
for file_name in uploaded.keys():
    with open(file_name, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            resume_text += page.extract_text()

# Job Description (Edit this)
job_desc = """
Looking for Python, Machine Learning, Data Analysis, SQL, Communication skills.
"""

# Convert text to vectors
cv = CountVectorizer(stop_words='english')
vectors = cv.fit_transform([resume_text, job_desc])

# Similarity score
similarity = cosine_similarity(vectors)[0][1]
score = similarity * 100

# Keywords extraction
resume_words = set(resume_text.lower().split())
job_words = set(job_desc.lower().split())

matched = resume_words & job_words
missing = job_words - resume_words

# Output
print(f"\nATS Match Score: {score:.2f}%")

print("\nMatched Keywords:")
print(matched)

print("\nMissing Keywords:")
print(missing)
