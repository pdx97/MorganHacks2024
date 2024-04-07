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
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()  # For a nicer styling of the plots
import matplotlib.pyplot as plt




# Load the resume dataset
resumeDataSet = pd.read_csv('C:/Users/prakh/Desktop/MorganHacks/data/resumeDataSet.csv', encoding='utf-8')

# Define the function to extract skills
def extract_skills(resumeText):
    skills_list = [
        'python', 'java', 'sql', 'excel', 'machine learning', 'project management',
        'javascript', 'html', 'css', 'react', 'node.js', 'angular', 'vue.js', 'php',
        'ruby on rails', 'c++', 'c#', '.net', 'swift', 'kotlin', 'android development',
        'ios development', 'flutter', 'docker', 'kubernetes', 'aws', 'azure', 'gcp',
        'big data', 'hadoop', 'spark', 'nosql', 'mongodb', 'cassandra', 'firebase',
        'data analysis', 'data visualization', 'tableau', 'power bi', 'sap', 'erp',
        'crm', 'seo', 'sem', 'ppc', 'email marketing', 'social media marketing', 'content creation',
        'graphic design', 'adobe photoshop', 'illustrator', 'indesign', 'ui/ux design',
        'cybersecurity', 'ethical hacking', 'penetration testing', 'network security',
        'blockchain', 'smart contracts', 'solidity', 'ethereum', 'financial modeling',
        'quantitative analysis', 'risk management', 'statistical analysis', 'r', 'matlab',
        'supply chain management', 'logistics', 'inventory management', 'autocad',
        'solidworks', '3d modeling', 'plc programming', 'electrical engineering', 'mechanical engineering',
        'civil engineering', 'biotechnology', 'molecular biology', 'genetics', 'clinical research',
        'nursing', 'pharmacy', 'public health', 'educational technology', 'curriculum development',
        'teaching', 'counseling', 'psychology', 'sociology', 'political science', 'international relations'
    ]

    extracted_skills = []
    for skill in skills_list:
        escaped_skill = re.escape(skill)
        if re.search(escaped_skill, resumeText, re.IGNORECASE):
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
train_duration_seconds = 10

pbar = tqdm(total=train_duration_seconds, desc="Training Progress")
clf.fit(X_train, y_train)
for _ in range(train_duration_seconds):
    time.sleep(1)
    pbar.update(1)
pbar.close()

# Making predictions
predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy*100:.2f}%")

# Calculate the confusion matrix
cm = confusion_matrix(y_test, predictions)

# Plotting
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# Adjust the labels if you have specific names for classes
class_names = le.classes_  # Assuming 'le' is your LabelEncoder object
plt.xticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names, rotation=45, ha="right")
plt.yticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names, rotation=45, va="center")
plt.show()


# Updated job category skills dictionary with skills for each category
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
    'Environmental Science': ['environmental policy', 'sustainability strategies', 'GIS', 'data analysis', 'ecology', 'environmental impact assessment'],
    'Arts': ['painting', 'sculpting', 'drawing', 'photography', 'graphic design', 'art history']
}


# Recommend job categories based on skills similarity
def recommend_job_categories(sample_skills_vector, top_n=3):
    similarity_scores = cosine_similarity(sample_skills_vector, X_train).flatten()
    top_indices = np.argsort(similarity_scores)[-top_n:]
    top_categories_encoded = y_train[top_indices]
    top_categories = le.inverse_transform(top_categories_encoded)
    return top_categories

# Find skills to learn for a recommended job category
def find_skills_to_learn(recommended_category, user_skills):
    user_skills_list = user_skills.split(', ')
    required_skills = job_category_skills.get(recommended_category, [])
    skills_to_learn = [skill for skill in required_skills if skill not in user_skills_list]
    return skills_to_learn

def display_predicted_categories(sample_index):
    sample_skills = resumeDataSet.loc[sample_index, 'skills']
    sample_category_encoded = predictions[sample_index]
    sample_category = le.inverse_transform([sample_category_encoded])[0]
    print(f"Skills: {sample_skills}")
    print(f"Predicted Job Category: {sample_category}")

def predict_and_recommend(sample_resume_text):
    # Extract skills from user's resume
    user_skills = extract_skills(sample_resume_text)

    # Vectorize user skills
    user_skills_vector = tfidf.transform([user_skills])

    # Make prediction
    predicted_category_encoded = clf.predict(user_skills_vector)[0]
    predicted_category = le.inverse_transform([predicted_category_encoded])[0]

    # Recommend job categories based on similarity
    recommended_categories = recommend_job_categories(user_skills_vector, top_n=3)

    # Find skills to learn for recommended categories
    skills_to_learn = {}
    for category in recommended_categories:
        skills_to_learn[category] = find_skills_to_learn(category, user_skills)

    # Display prediction results
    print("Predicted Job Category:", predicted_category)
    print("Recommended Job Categories:", recommended_categories)
    for category, skills in skills_to_learn.items():
        print(f"Skills to learn for {category}:", skills)

# Example: Predict and recommend for a sample resume
sample_resume_text = resumeDataSet.loc[56, 'Resume']
predict_and_recommend(sample_resume_text)
