# Import necessary libraries
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the resume dataset
resumeDataSet = pd.read_csv('C:/Users/prakh/Desktop/MorganHacks/data/resumeDataSet.csv', encoding='utf-8')

# Define a basic list of skills for demonstration
skills_list = [
    'python', 'java', 'sql', 'excel', 'machine learning', 'project management',
    'c/c++', 'javascript', 'html/css', 'git', 'linux/unix', 'matlab/r',
    'aws/azure/gcp', 'docker', 'kubernetes', 'restful api development',
    'data structures and algorithms', 'network security',
    'deep learning', 'natural language processing', 'computer vision',
    'data mining', 'data analysis', 'statistical modeling', 'time series analysis',
    'big data technologies', 'tensorflow', 'pytorch', 'scikit-learn',
    'database management systems', 'data warehousing', 'etl',
    'data visualization', 'front-end development', 'back-end development',
    'full-stack development', 'web frameworks', 'responsive design',
    'api integration', 'cross-browser compatibility', 'web security',
    'communication skills', 'leadership', 'problem-solving', 'time management',
    'teamwork and collaboration', 'adaptability', 'creativity', 'critical thinking',
    'financial analysis', 'healthcare informatics', 'e-commerce platforms',
    'digital marketing analytics', 'supply chain management', 'cybersecurity',
    'renewable energy technologies', 'mobile application development',
    'game development'
]

# Function to extract skills from resume texts
def extract_skills(resumeText):
    skills = [skill for skill in skills_list if skill in resumeText.lower()]
    return ', '.join(skills)

# Apply skills extraction
resumeDataSet['skills'] = resumeDataSet['Resume'].apply(extract_skills)

# Define features and target
X = resumeDataSet['skills']
y = resumeDataSet['Category']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the skills using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initialize and train the classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_vec, y_train)

# Make predictions
predictions = classifier.predict(X_test_vec)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)# Function to display predicted job categories for a sample resume


def display_predicted_categories(sample_index):
    sample_skills = resumeDataSet.loc[sample_index, 'skills']
    sample_category_encoded = predictions[sample_index]
    sample_category = le.inverse_transform([sample_category_encoded])[0]
    user_current_job_role_encoded = y_test[sample_index]
    user_current_job_role = le.inverse_transform([user_current_job_role_encoded])[0]

    print(f"Skills: {sample_skills}")
    print(f"Predicted Job Category: {sample_category}")
    # print(f"User's Current Job Role: {user_current_job_role}")



# Display predictions for the first resume in the test set
sample_index = 2  # Adjust as needed to show other examples
display_predicted_categories(sample_index)

from sklearn.metrics.pairwise import cosine_similarity

# Function to recommend job categories based on skills similarity
def recommend_job_categories(sample_index, top_n=3):
    sample_resume_vector = X_test[sample_index]

    # Calculate similarity scores between the sample resume and all categories
    similarity_scores = cosine_similarity(sample_resume_vector, X_train).flatten()

    # Get top N matching categories based on similarity scores
    top_indices = np.argsort(similarity_scores)[-top_n:]
    top_categories_encoded = y_train[top_indices]
    top_categories = le.inverse_transform(top_categories_encoded)

    return top_categories

# Recommend job categories for the first resume in the test set
recommended_categories = recommend_job_categories(sample_index, top_n=3)
print("Recommended Job Categories:", recommended_categories)

import matplotlib.pyplot as plt

# Function to calculate confidence scores for recommended job categories
def calculate_confidence_scores(sample_index, top_n=3):
    sample_resume_vector = X_test[sample_index]

    # Calculate similarity scores between the sample resume and all categories
    similarity_scores = cosine_similarity(sample_resume_vector, X_train).flatten()

    # Get top N matching categories based on similarity scores
    top_indices = np.argsort(similarity_scores)[-top_n:]
    top_categories_encoded = y_train[top_indices]
    top_categories = le.inverse_transform(top_categories_encoded)

    # Get confidence scores
    confidence_scores = [similarity_scores[index] for index in top_indices]

    return top_categories, confidence_scores

# Calculate confidence scores for the first resume in the test set
recommended_categories, confidence_scores = calculate_confidence_scores(sample_index, top_n=3)

# Build confidence matrix
confidence_matrix = pd.DataFrame({'Job Category': recommended_categories, 'Confidence Score': confidence_scores})

# Plot the confidence scores
plt.figure(figsize=(10, 6))
plt.barh(confidence_matrix['Job Category'], confidence_matrix['Confidence Score'], color='skyblue')
plt.xlabel('Confidence Score')
plt.title('Confidence Scores for Recommended Job Categories')
plt.gca().invert_yaxis()  # Invert y-axis to display the highest score on top
plt.show()


# Evaluate the model (This step requires you to calculate the model's performance using metrics such as accuracy, precision, recall, etc.)
# metrics.accuracy_score(y_test, predictions)
# metrics.precision_score(y_test, predictions, average='macro')
# metrics.recall_score(y_test, predictions, average='macro')

# Add your evaluation code here
