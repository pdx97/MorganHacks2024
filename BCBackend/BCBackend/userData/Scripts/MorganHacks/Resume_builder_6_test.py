import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib

# Define the path to the CSV file
file_path = 'C:/Users/prakh/Desktop/MorganHacks/data/resumeDataSet.csv'

# Load the CSV file
user_skills_df = pd.read_csv(file_path)
# print(user_skills_df.columns)

# Function to clean the text data
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    text = text.strip()
    return text

# Apply the cleaning function to the resume text
user_skills_df['cleaned_resume'] = user_skills_df['Resume'].apply(clean_text)

# Load the TF-IDF vectorizer and KMeans model
# Note: The TF-IDF vectorizer should be fitted with the dataset or loaded from a previously saved state
# if you want to maintain consistency with previously vectorized text.
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(user_skills_df['cleaned_resume'])

# Load the trained KMeans model
# model_weights_path = 'data/kmeans_model_weights_10_clusters.pkl'
model_weights_path = 'kmeans_model_15_clusters.pkl'
kmeans = joblib.load(model_weights_path)

# Function to predict the job category
def predict_job_category(resume_text):
    cleaned_text = clean_text(resume_text)
    vectorized_text = tfidf_vectorizer.transform([cleaned_text])
    predicted_cluster = kmeans.predict(vectorized_text)
    return predicted_cluster[0]

# Load the user's skills CSV file
user_input_path = 'C:/Users/prakh/Desktop/MorganHacks/data/user_skills_sample_other.csv'
user_input_df = pd.read_csv(user_input_path)

# Predict the job category for the user's skills
user_skills = user_input_df['Skills'].iloc[0]
predicted_job_category = predict_job_category(user_skills)

# Define a hypothetical mapping of cluster numbers to job categories
cluster_to_job_category = {
    0: "Administrative",
    1: "Management",
    2: "Customer Service",
    3: "Software Development",
    4: "Data Science",
    5: "Gig Worker - Delivery",
    6: "Gig Worker - Ride Share",
    7: "Creative Arts",
    8: "Healthcare",
    9: "Education",
    10: "Technical Writing",
    11: "Project Coordination",
    12: "Digital Marketing",
    13: "Human Resources",
    14: "Legal Services"
}
# Print the predicted job category
print(f"The predicted job category for the skills '{user_skills}' is: {cluster_to_job_category[predicted_job_category]}")
