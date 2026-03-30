# Install required library
!pip install scikit-learn

# Import libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample Resume
resume = """
I am a BTech student in Artificial Intelligence and Data Science.
Skilled in Python, Machine Learning, Data Analysis, Deep Learning.
"""

# Sample Job Description
job_desc = """
Looking for a candidate with Python, Machine Learning, Data Analysis, SQL, and Communication skills.
"""

# Convert text to vectors
cv = CountVectorizer()
vectors = cv.fit_transform([resume, job_desc])

# Calculate similarity
similarity = cosine_similarity(vectors)[0][1]

# Convert to percentage
score = similarity * 100

# Find keywords
resume_words = set(resume.lower().split())
job_words = set(job_desc.lower().split())

missing = job_words - resume_words

# Output
print(f"ATS Match Score: {score:.2f}%")
print("\nMissing Keywords:")
print(missing)
