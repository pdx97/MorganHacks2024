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

# Load the resume dataset from CSV
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
train_duration_seconds = 2

pbar = tqdm(total=train_duration_seconds, desc="Training Progress")
clf.fit(X_train, y_train)
for _ in range(train_duration_seconds):
    time.sleep(1)
    pbar.update(1)
pbar.close()

# Making predictions
predictions = clf.predict(X_test)

job_category_skills = {
    'Data Science': ['python', 'machine learning', 'data analysis', 'statistics', 'sql', 'deep learning'],
    'Human Resources (HR)': ['recruitment', 'employee relations', 'compliance', 'payroll management', 'performance management', 'HRIS software', 'training and development', 'benefits administration'],
    'Advocate': ['legal research', 'litigation', 'client counseling', 'contract drafting', 'compliance', 'case preparation', 'public speaking', 'legal writing'],
    'Arts': ['creative skills', 'art history', 'visual design', 'photography', 'sculpting', 'painting', 'drawing', 'exhibition planning'],
    'Web Designing': ['html', 'css', 'javascript', 'graphic design', 'responsive design', 'SEO', 'CMS', 'user experience design'],
    'Mechanical Engineering': ['CAD', 'CAM', 'thermodynamics', 'fluid mechanics', 'material science', 'mechanical systems', 'project management', 'quality assurance'],
    'Sales': ['customer relationship management', 'negotiation', 'product knowledge', 'sales strategy', 'market analysis', 'communication skills', 'presentation skills', 'CRM software'],
    'Health and Fitness': ['personal training', 'nutrition planning', 'exercise physiology', 'group fitness', 'wellness coaching', 'injury prevention', 'first aid', 'fitness assessments'],
    'Civil Engineering': ['structural engineering', 'project management', 'construction management', 'urban planning', 'environmental engineering', 'geotechnical engineering', 'surveying', 'CAD software'],
    'Java Developer': ['java', 'spring framework', 'hibernate', 'microservices', 'API development', 'Maven/Gradle', 'JUnit', 'database management'],
    'Business Analyst': ['business intelligence', 'requirements analysis', 'data analysis', 'project management', 'stakeholder management', 'SQL', 'process modeling', 'agile methodologies'],
    'SAP Developer': ['SAP modules', 'ABAP programming', 'SAP HANA', 'data modeling', 'business process knowledge', 'SAP Fiori', 'system integration', 'troubleshooting'],
    'Automation Testing': ['test automation frameworks', 'Selenium', 'scripting', 'CI/CD', 'version control', 'bug tracking', 'performance testing', 'agile methodologies'],
    'Electrical Engineering': ['circuit analysis', 'power systems', 'PLC programming', 'electrical design', 'CAD software', 'control systems', 'safety standards', 'troubleshooting'],
    'Operations Manager': ['operations management', 'supply chain management', 'budgeting', 'process improvement', 'team leadership', 'logistics', 'inventory management', 'strategic planning'],
    'Python Developer': ['python', 'Django/Flask', 'RESTful API development', 'SQL/NoSQL databases', 'object-oriented programming', 'unit testing', 'version control', 'debugging'],
    'DevOps Engineer': ['CI/CD pipelines', 'containerization', 'Kubernetes', 'cloud services', 'scripting', 'monitoring tools', 'configuration management', 'security best practices'],
    'Network Security Engineer': ['network security', 'firewalls', 'intrusion detection systems', 'VPN', 'incident response', 'security protocols', 'ethical hacking', 'compliance standards'],
    'PMO': ['project management', 'program management', 'portfolio management', 'PMO governance', 'resource allocation', 'budgeting', 'risk management', 'reporting'],
    'Database': ['SQL', 'database design', 'data modeling', 'performance tuning', 'backup and recovery', 'NoSQL databases', 'data warehousing', 'ETL processes'],
    'Hadoop': ['Hadoop ecosystem', 'MapReduce', 'HDFS', 'Spark', 'big data analytics', 'Hive', 'Pig', 'data ingestion and processing'],
    'ETL Developer': ['ETL tools', 'data warehousing', 'data modeling', 'SQL', 'scripting', 'data quality', 'database performance', 'business intelligence'],
    'DotNet Developer': ['.NET framework', 'C#', 'ASP.NET MVC', 'Entity Framework', 'WCF/WPF', 'SQL Server', 'Visual Studio', 'version control'],
    'Blockchain': ['blockchain technology', 'smart contracts', 'Ethereum', 'consensus algorithms', 'cryptography', 'solidity', 'distributed ledger', 'node.js'],
    'Testing': ['test case development', 'manual testing', 'automated testing', 'bug tracking', 'performance testing', 'security testing', 'test planning', 'quality assurance']
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

# Example: Predict and recommend for the first resume in the dataset
sample_resume_text = resumeDataSet.loc[3, 'Resume']
predict_and_recommend(sample_resume_text)
