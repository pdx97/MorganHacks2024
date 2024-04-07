# Import necessary libraries
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import time

# Load the resume dataset
resumeDataSet = pd.read_csv('C:/Users/prakh/Desktop/MorganHacks/data/resumeDataSet.csv', encoding='utf-8')

# Define the function to extract skills
def extract_skills(resumeText):
    skills_list = ['python', 'java', 'sql', 'excel', 'machine learning', 'project management']
    extracted_skills = []
    for skill in skills_list:
        if re.search(skill, resumeText, re.IGNORECASE):
            extracted_skills.append(skill)
    return ', '.join(extracted_skills)

# Apply skills extraction
resumeDataSet['skills'] = resumeDataSet['Resume'].apply(extract_skills)

# Label encoding
le = LabelEncoder()
resumeDataSet['Category'] = le.fit_transform(resumeDataSet['Category'])

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=1500)
X = tfidf.fit_transform(resumeDataSet['skills'].values)
y = resumeDataSet['Category'].values

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





# Classifier Initialization
clf = OneVsRestClassifier(KNeighborsClassifier())

# Training with a progress bar
train_duration_seconds = 2

pbar = tqdm(total=train_duration_seconds, desc="Training Progress")
clf.fit(X_train, y_train)
for _ in range(train_duration_seconds):
    time.sleep(1)
    pbar.update(1)
pbar.close()

# Making predictions
predictions = clf.predict(X_test)

from sklearn.metrics import accuracy_score

# Predict the accuracy of the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy of the Model: {accuracy}")



job_category_skills = {
    'Data Science': ['python', 'machine learning', 'data analysis', 'statistics', 'sql', 'deep learning'],
    'Engineering': ['project management', 'java', 'c++', 'python', 'autocad', 'matlab'],
    'Web Development': ['javascript', 'html', 'css', 'react', 'node.js', 'sql'],
    'Plumber': ['plumbing repair', 'installation', 'piping and fixtures knowledge', 'welding', 'customer service'],
    'Barber': ['hair cutting techniques', 'shaving', 'styling', 'hair care products knowledge', 'customer service'],
    'Graphic Design': ['Adobe Creative Suite', 'branding', 'typography', 'UI/UX design', 'print design', 'web design'],
    'Digital Marketing': ['SEO/SEM', 'content marketing', 'social media management', 'email marketing', 'Google Analytics', 'PPC advertising'],
    'Finance': ['financial analysis', 'accounting', 'Excel', 'budgeting', 'financial modeling', 'investment management'],
    'Education': ['curriculum development', 'classroom management', 'educational technology', 'assessment design', 'special education', 'online teaching'],
    'Healthcare': ['patient care', 'medical knowledge', 'EMR systems', 'healthcare laws and ethics', 'emergency response', 'medical research'],
    'Real Estate': ['property management', 'sales and negotiation', 'real estate laws', 'market analysis', 'customer service', 'property valuation'],
    'Hospitality': ['customer service', 'event planning', 'food and beverage management', 'hospitality management', 'tourism knowledge', 'multilingual'],
    'Retail': ['customer service', 'sales expertise', 'inventory management', 'visual merchandising', 'product knowledge', 'POS systems'],
    'Cybersecurity': ['network security', 'ethical hacking', 'information assurance', 'malware analysis', 'firewall administration', 'cryptographic skills'],
    'Environmental Science': ['environmental policy', 'sustainability strategies', 'GIS', 'data analysis', 'ecology', 'environmental impact assessment']
}

# Recommend job categories based on skills similarity
def recommend_job_categories(sample_index, top_n=3):
    sample_resume_vector = X_test[sample_index]
    similarity_scores = cosine_similarity(sample_resume_vector, X_train).flatten()
    top_indices = np.argsort(similarity_scores)[-top_n:]
    top_categories_encoded = y_train[top_indices]
    top_categories = le.inverse_transform(top_categories_encoded)
    return top_categories

# Find skills to learn for a recommended job category
def find_skills_to_learn(recommended_category, user_skills, job_category_skills):
    user_skills_list = user_skills.split(', ')
    required_skills = job_category_skills.get(recommended_category, [])
    skills_to_learn = [skill for skill in required_skills if skill not in user_skills_list]
    return skills_to_learn

def display_predicted_categories(sample_index):
    sample_skills = resumeDataSet.loc[sample_index, 'skills']
    sample_category_encoded = predictions[sample_index]
    sample_category = le.inverse_transform([sample_category_encoded])[0]

    # print(f"Resume Index: {sample_index}")
    print(f"Skills: {sample_skills}")
    print(f"Predicted Job Category: {sample_category}")

# Example: Predict and recommend for the first resume in the test set
sample_index = 19








# Display the user's current job category based on the sample index
user_category_encoded = y_test[sample_index]
user_category = le.inverse_transform([user_category_encoded])[0]
print(f"User's Current Job Category: {user_category}")



display_predicted_categories(sample_index)
recommended_categories = recommend_job_categories(sample_index)
print("Recommended Job Categories:", recommended_categories)

# Assuming 'Data Science' was recommended, find skills to learn
user_skills = resumeDataSet.loc[sample_index, 'skills']
skills_to_learn = find_skills_to_learn('Data Science', user_skills, job_category_skills)
print("Skills to learn for Data Science:", skills_to_learn)
