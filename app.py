import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pickle
import time

# Custom Transformer for dropping columns
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors='ignore')

# Custom Transformer for filling NaN values
class NaNFiller(BaseEstimator, TransformerMixin):
    def __init__(self, fill_value="NA"):
        self.fill_value = fill_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.fillna(self.fill_value)


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.markdown("""
    <style>
    .custom-title {
        top: 0;
        left: 0;
        right: 0;
        text-align: center;
        font-size: 50px;
        font-style: italic;
        font-weight: bold;
        color: #187ecc;
        z-index: 9999;
        background: None;
        padding: 10px 0;
    }
    </style>
    <div class='custom-title'>Salary Prediction Tool</div>
    """, unsafe_allow_html=True)



with st.container(border=True):
    st.header(":red[Abstract]")
    st.markdown("""To ensure that there is no discrimination among employees, it is imperative that the
    Human Resources (HR) department of Company X maintains a consistent salary range
    for employees with similar profiles. Apart from the existing salary, various other factors—
    such as experience and other assessed abilities—play a role in salary decisions. This
    project aims to build a predictive model that determines the salary to be offered to a
    selected candidate, thereby reducing human bias in the salary negotiation process. """)

with st.container(border=True):
    st.header(":red[Goal and Objective]")
    st.markdown("""The objective of this project is to build a regression model using historical hiring data
    to predict the expected salary of a candidate. This model aims to minimize manual
    judgment and potential bias, ensuring fairness and transparency in salary decisions for
    employees with similar profiles.""")

departments = ("IT-Software", "Accounts", "Top Management", "Engineering", "Education", "Banking", "HR", "Sales", "Healthcare", "Analytics/BI", "Marketing", "Others")
roles = ('Consultant', 'Financial Analyst', 'Project Manager', 'Area Sales Manager', 'Team Lead', 'Analyst', 'CEO', 'Business Analyst', 'Sales Manager', 'Bio statistician', 'Scientist', 'Research Scientist', 'Head', 'Associate', 'Senior Researcher', 'Sales Execituve', 'Sr. Business Analyst', 'Principal Analyst', 'Data scientist', 'Researcher', 'Senior Analyst', 'Professor', 'Lab Executuve', "Others")
industries = ('Analytics', 'Training', 'Aviation', 'Insurance', 'Retail', 'FMCG', 'Telecom', 'Automobile', 'IT', 'BFSI', 'Others')
designations = ('HR', 'Medical Officer', 'Director', 'Marketing Manager', 'Manager', 'Product Manager', 'Consultant', 'CA', 'Research Scientist', 'Sr.Manager', 'Data Analyst', 'Assistant Manager', 'Web Designer', 'Research Analyst', 'Software Developer', 'Network Engineer', 'Scientist', 'Others')
educations = ('PG', 'Doctorate', 'Grad', 'Under Grad')
specializations = ('NA','Arts', 'Chemistry', 'Zoology', 'Sociology', 'Psychology', 'Mathematics', 'Engineering', 'Botony', 'Statistics', 'Economics', 'Others')
pg_specs = ('NA','Zoology', 'Chemistry', 'Psychology', 'Mathematics', 'Engineering', 'Sociology', 'Arts', 'Statistics', 'Economics', 'Botony', "Others")
phd_specs = ('NA','Chemistry', 'Zoology', 'Psychology', 'Engineering', 'Botony', 'Arts', 'Statistics', 'Economics', 'Mathematics', 'Sociology', 'Others')
ratings = ('NA', 'Key_Performer', 'A', 'B', 'C', 'D')
offers = ('Y', ' N')

with st.form("data_form"):
    st.markdown("Inorder to predict the expected salary of a given candidate please enter the following details :")
    st.markdown(":green[***Job Details :***]",)
    applicant_id = st.number_input("Applicant ID :", step = 1)
    total_experience = st.number_input("Total Experience :", step = 1)
    total_experience_in_field = st.number_input("Total Experience in the applied field:", step = 1)
    department = st.selectbox("Department :", departments)
    role = st.selectbox("Role :", roles)
    industry = st.selectbox("Industry", industries)
    designation = st.selectbox("Designation :", designations)

    st.markdown(":green[***Educational Details :***]",)
    education = st.selectbox("Eduction Level :", educations)
    pass_year_grad = st.number_input("Enter Passing Year of Graduation :",min_value= 1900, max_value=3000,  step = 1)
    graduation_specialization = st.selectbox("Graduation Specialization :", specializations)
    passing_year_pg = st.number_input("Enter Passing Year of PG :",min_value= 1900, max_value=3000,  step = 1)
    pg_spec = st.selectbox("PG Specialization :", pg_specs)
    passing_year_phd = st.number_input("Enter Passing Year of PHD :",min_value= 1900, max_value=3000,  step = 1)
    phd_spec = st.selectbox("PHD Specialization :", phd_specs)
    certifications = st.number_input("Certifications :", min_value=0, step= 1)
    publications = st.number_input("Number of Publications :", min_value=0, step= 1)
    inter_degree = st.number_input("International Degrees obtained:", min_value=0, step=1)

    st.markdown(":green[***Previous Occupation Details :***]")
    last_appraisal_rating = st.selectbox("Last Appraisal Rating :", ratings)
    inhand_offer = st.selectbox("Inhand offer :", offers)
    current_ctc = st.number_input("Current CTC :", min_value = 0, step = 1)
    no_companies = st.number_input("Number of Companies worked at till now :",min_value=0, step =1)
    
    submitted = st.form_submit_button("Submit")
    
if submitted:
    data = pd.DataFrame([{
        "index": 0,
        "IDX": 0,
        "Applicant_ID": applicant_id,
        "Total_Experience": total_experience,
        "Total_Experience_in_field_applied": total_experience_in_field,
        "Department": department,
        "Role": role,
        "Industry": industry,
        "Organization": "NA",
        "Designation":designation,
        "Education": education,
        "Graduation_Specialization": graduation_specialization,
        "University_Grad": "NA",
        "Passing_Year_Of_Graduation": pass_year_grad,
        "PG_Specialization": pg_spec,
        "University_PG": "NA",
        "Passing_Year_Of_PG": passing_year_pg,
        "PHD_Specialization": phd_spec,
        "University_PHD": "NA",
        "Passing_Year_Of_PHD": passing_year_phd,
        "Curent_Location": "NA",
        "Preferred_location": "NA",
        "Current_CTC": current_ctc,
        "Inhand_Offer": inhand_offer,
        "Last_Appraisal_Rating": last_appraisal_rating,
        "No_Of_Companies_worked": no_companies,
        "Number_of_Publications": publications,
        "Certifications": certifications,
        "International_degree_any": inter_degree
        }])
    expected_ctc = model.predict(data)
    st.success(f"The estimated CTC to be provided is Rs. {round(expected_ctc[0]):,} per annum.")

st.markdown("""
    <hr style="margin-top: 50px;">
    <div style="text-align: center; font-size: 14px; color: gray;">
        Developed by <strong>Helem Thekkumvilayil Jose</strong>  
        <br>B.Tech in Electrical Engineering, IIT (ISM) Dhanbad  
        <br>Machine Learning Capstone Project, June 2025
    </div>
    """, unsafe_allow_html=True)