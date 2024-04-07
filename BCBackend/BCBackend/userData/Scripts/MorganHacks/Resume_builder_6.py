import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib

# Load the CSV file
file_path = 'C:/Users/prakh/Desktop/MorganHacks/data/resumeDataSet.csv'
user_skills_df = pd.read_csv(file_path)

# Clean the text data
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    text = text.strip()
    return text

# Apply cleaning
user_skills_df['cleaned_resume'] = user_skills_df['Resume'].apply(clean_text)

# Initialize and fit the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(user_skills_df['cleaned_resume'])

# Initialize and fit the KMeans model with increased number of clusters
# Adjust the number of clusters as needed to better match the diversity of job categories
kmeans = KMeans(n_clusters=10, random_state=42)  # Increased number of clusters
kmeans.fit(tfidf_matrix)

# Define a new mapping for clusters to job categories, including categories for gig workers
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
    9: "Education"
}

# Predict the job category for an example resume
def predict_job_category(resume_text):
    cleaned_text = clean_text(resume_text)
    vectorized_text = tfidf_vectorizer.transform([cleaned_text])
    predicted_cluster = kmeans.predict(vectorized_text)
    return cluster_to_job_category[predicted_cluster[0]]

# Example usage
example_resume_text = user_skills_df['Resume'].iloc[1]
predicted_job_category = predict_job_category(example_resume_text)
print(f"Predicted Job Category: {predicted_job_category}")


# Save the model weights (cluster centers) to a file
model_weights_path = 'C:/Users/prakh/Desktop/MorganHacks/data/kmeans_model_weights_10_clusters.pkl'
joblib.dump(kmeans, model_weights_path)

# Output the path to the saved model weights
print(f"The model weights have been saved to: {model_weights_path}")

#
# # Let's define a function that predicts job categories based on the user's resume using the trained KMeans model.
# # For simplicity, we'll assume that each cluster corresponds to a different job category.
#
# def predict_job_category(resume_text):
#     # Clean the input resume text
#     cleaned_text = clean_text(resume_text)
#
#     # Vectorize the cleaned resume text using the fitted TF-IDF vectorizer
#     vectorized_text = tfidf_vectorizer.transform([cleaned_text])
#
#     # Use the trained KMeans model to predict the cluster
#     predicted_cluster = kmeans.predict(vectorized_text)
#
#     return predicted_cluster[0]
#
# # Predict the job category for the first resume in the dataset as an example
# example_resume_text = user_skills_df['Resume'].iloc[1]
# predicted_job_category = predict_job_category(example_resume_text)
#
# predicted_job_category
#
def get_top_skills_for_each_category(n_terms=10):
    terms = tfidf_vectorizer.get_feature_names_out()
    category_skills = {}
    for i in range(kmeans.n_clusters):
        top_terms_indices = kmeans.cluster_centers_.argsort()[:, -n_terms:][i, :]
        top_terms = [terms[ind] for ind in top_terms_indices]
        category_skills[i] = top_terms
    return category_skills

def get_top_10_jobs_for_resume(resume_text):
    vectorized_text = tfidf_vectorizer.transform([clean_text(resume_text)])
    distances = kmeans.transform(vectorized_text)
    top_clusters_indices = distances.argsort()[0]
    top_10_clusters = top_clusters_indices[:10]
    return top_10_clusters

category_skills = get_top_skills_for_each_category()
# example_resume_text = user_skills_df['Resume'].iloc[0]
top_10_jobs = get_top_10_jobs_for_resume(example_resume_text)

top_10_jobs, {cluster: category_skills[cluster] for cluster in top_10_jobs}


def print_predicted_categories_and_skills(resume_text):
    # Predict the cluster number directly
    cleaned_text = clean_text(resume_text)
    vectorized_text = tfidf_vectorizer.transform([cleaned_text])
    predicted_cluster = kmeans.predict(vectorized_text)[0]  # Directly use cluster number

    # Get the job category name from the cluster number
    predicted_job_category = cluster_to_job_category[predicted_cluster]

    # Retrieve top skills for the predicted cluster
    top_skills = category_skills[predicted_cluster]  # Use cluster number to access top skills

    # Print the predicted category and the skills required for that category
    print(f"Predicted Job Category: {predicted_job_category}")
    print(f"Skills required for Job Category {predicted_cluster}: {', '.join(top_skills)}\n")

# Print the predicted categories and the skills required for the top 3 jobs of the example resume
for job in top_10_jobs:
    print_predicted_categories_and_skills(user_skills_df['Resume'].iloc[job])
