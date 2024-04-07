# import streamlit as st
# import pandas as pd
# import joblib
# from sklearn.metrics.pairwise import linear_kernel
# import streamlit.components.v1 as components
#
# # Load the TF-IDF vectorizer and cosine similarity matrix
# tfidf = joblib.load("C:/Users/prakh/Downloads/tfidf_vectorizer.pkl")
# cosine_sim = joblib.load("C:/Users/prakh/Downloads/cosine_similarity.pkl")
#
# # Load the job dataset
# df = pd.read_csv('C:/Users/prakh/Downloads/dice_com-job_us_sample.csv')
# df = df.dropna().reset_index(drop=True)
# indices = pd.Series(df.index, index=df['jobtitle']).drop_duplicates()
#
# # Assuming a simple mapping of job titles to roadmap image paths
# # This is a placeholder. You will need to create a mapping based on your available images and job titles.
# # Map job titles to roadmap image paths. Adjust with actual data.
# job_images = {
#     'IT Security Engineer': [
#         r'C:\Users\prakh\Downloads\Application_security.jpg',
#         # Add paths to additional course images as needed
#     ],
#     'Application Security Engineer': [
#         r"C:\Users\prakh\Downloads\IT Career.png",
#         # Add paths to additional course images as needed
#     ],
#     # Add more mappings for other job titles as needed
#     'Business Analyst': [
#         r'C:\Users\prakh\Downloads\Business-analyst-careers-.jpg',
#         # Add paths to additional course images as needed
#     ],
#     'IT Security Engineer': [
#         r'C:\Users\prakh\Downloads\Application_security.jpg',
#         # Add paths to additional course images as needed
#     ],
# }
# # Define the job recommendation function
# def get_recommendation(title, cosine_sim=cosine_sim):
#     if title not in indices:
#         return pd.Series([])  # If title not found, return an empty series
#     idx = indices[title]
#     sim_scores = list(enumerate(cosine_sim[idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     sim_scores = sim_scores[1:4]  # Adjust this number for the number of recommendations
#     tech_indices = [i[0] for i in sim_scores]
#     recommended_jobs = df.iloc[tech_indices][['jobtitle', 'skills']]
#     return recommended_jobs
#
# # Create the Streamlit app
# def main():
#     st.title('Job Recommender System')
#
#     job_title = st.text_input('Enter a job title:', 'Lead DevOps Engineer')
#
#     if st.button('Get Job Recommendations'):
#         recommendations = get_recommendation(job_title)
#         if recommendations.empty:
#             st.write("Sorry, no recommendations found.")
#         else:
#             st.subheader('Recommended Jobs:')
#             for index, row in recommendations.iterrows():
#                 st.markdown(f"**{row['jobtitle']}**")
#                 st.markdown(f"*Key Skills Required*: {row['skills']}")
#                  # Display images associated with each job title
#                 if row['jobtitle'] in job_images:
#                     images = job_images[row['jobtitle']]
#                     for img_path in images:
#                         st.image(img_path, caption=f'Visual Guide for {row["jobtitle"]}',width=300)
#                 else:
#                     st.write("Visual guide not available for this job.")
#                 # # Displaying course images for each recommended job
#                 # for index, row in recommendations.iterrows():
#                 #     st.markdown(f"**{row['jobtitle']}**")
#                 #     st.markdown(f"*Key Skills Required*: {row['skills']}")
#
#
#
#
# if __name__ == '__main__':
#     main()


import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import linear_kernel

# Initialize session state for page navigation
if 'show_recommendation_page' not in st.session_state:
    st.session_state['show_recommendation_page'] = False

def show_landing_page():
    st.title("Welcome to Pivot Careers")
    st.write("Discover your next career opportunity with our AI-driven job recommendation engine.")

    # External link to a survey or additional information
    st.markdown(f"[Take our survey!]({'https://tally.so/r/nr68Ov'})")

    if st.button("Explore Job Recommendations"):
        st.session_state['show_recommendation_page'] = True

# Load the TF-IDF vectorizer and cosine similarity matrix
tfidf = joblib.load("C:/Users/prakh/Downloads/tfidf_vectorizer.pkl")
cosine_sim = joblib.load("C:/Users/prakh/Downloads/cosine_similarity.pkl")

# Load the job dataset
df = pd.read_csv('C:/Users/prakh/Downloads/dice_com-job_us_sample.csv')
df = df.dropna().reset_index(drop=True)
indices = pd.Series(df.index, index=df['jobtitle']).drop_duplicates()

# Map job titles to roadmap image paths. Adjust with actual data.
job_images = {
    'IT Security Engineer': [r'C:\Users\prakh\Downloads\Application_security.jpg'],
    'Application Security Engineer': [r"C:\Users\prakh\Downloads\IT Career.png"],
    'Business Analyst': [r'C:\Users\prakh\Downloads\Business-analyst-careers-.jpg'],
    # Add more mappings for other job titles as needed
}

def get_recommendation(title, cosine_sim=cosine_sim):
    if title not in indices:
        return pd.DataFrame([])  # If title not found, return an empty DataFrame
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]  # Adjust this number for the number of recommendations
    tech_indices = [i[0] for i in sim_scores]
    return df.iloc[tech_indices][['jobtitle', 'skills']]

def main():
    st.title('Job Recommender System')
    job_title = st.text_input('Enter a job title:', 'Lead DevOps Engineer')

    if st.button('Get Job Recommendations'):
        recommendations = get_recommendation(job_title)
        if recommendations.empty:
            st.write("Sorry, no recommendations found.")
        else:
            st.subheader('Recommended Jobs:')
            for index, row in recommendations.iterrows():
                st.markdown(f"**{row['jobtitle']}**")
                st.markdown(f"*Key Skills Required*: {row['skills']}")
                # Display images associated with each job title
                if row['jobtitle'] in job_images:
                    images = job_images[row['jobtitle']]
                    for img_path in images:
                        st.image(img_path, caption=f'Visual Guide for {row["jobtitle"]}', width=300)
                else:
                    st.write("Visual guide not available for this job.")

# Decide which page to display
if __name__ == '__main__':
    if st.session_state['show_recommendation_page']:
        main()
    else:
        show_landing_page()
