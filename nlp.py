# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import numpy as np
import pickle
import pandas as pd
import docx2txt
import pdfplumber
import docx
import os
from docx import Document
import PyPDF2
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords')
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop=set(stopwords.words('english'))
from wordcloud import WordCloud
import plotly.express as px

lemmatizer = WordNetLemmatizer()
loaded_model=pickle.load(open(r"C:\Users\vk803\OneDrive\Desktop\project 3\model.pkl",'rb'))
vector=pickle.load(open(r"C:\Users\vk803\OneDrive\Desktop\project 3\vector.pkl",'rb'))

    

resume = []
def display(doc_file):
    if doc_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume.append(docx2txt.process(doc_file))
    else :
        with pdfplumber.open(doc_file) as pdf:
            pages=pdf.pages[0]
            resume.append(pages.extract_text())
    return resume



def cleanResume(resumeText):
    resumeText=str(resumeText)
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-/:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

def preprocess(text):
    text=text.lower()
    tokenize=word_tokenize(text)
    tokens=[lemmatizer.lemmatize(word) for word in tokenize if len(word)>2 if word and word.isalnum() not in stop]
    return " ".join(tokens)


def display_wordcloud(mostcommon):
    custom_stopwords = set(stop)
    custom_stopwords.update(['good', 'great', 'excellent', 'proficient', 'skill', 'skills','experience','workday','peoplesoft','project','application','knowledge','report','development','developed'])  # Add more stopwords as needed
    wordcloud=WordCloud(width=1000, height=600, background_color='black',stopwords=custom_stopwords).generate(str(mostcommon))
    a=px.imshow(wordcloud)
    st.plotly_chart(a)

def extract_skills(text):
    skills=[]
    specific_skills = [
       'react js', 'python' , 'javascript', 'html', 'css', 'webpack', 'npm',
        'hcm', 'report writer', 'eib', 'core connector',
       'sql', 'database', 'pl/sql', 'oracle', 'mysql', 't-sql'
       , 'hrms', 'peoplecode', 'application engine', 'sqr','administrator','xml','github','json','app package', 'applicationengine', 'peoplecode', 'sqr','bip', 'ps', 'query']

    for keyword in specific_skills:
        if keyword in text.lower():
            skills.append(keyword)
    return skills
 


def main():
    st.title('RESUME CLASSIFICATION')
    upload_file = st.file_uploader('Hey,Upload Your Resume ',
                                type= ['docx','pdf','doc'],accept_multiple_files=True)
    if st.button("Process"):
        for doc_file in upload_file:
            if doc_file is not None:
                file_details = {'filename':[doc_file.name],
                               'filetype':doc_file.type.split('.')[-1],
                               'filesize':str(doc_file.size)+' KB'}
                file_type=pd.DataFrame(file_details)
                st.write(file_type.set_index('filename'))
                
                text = display(doc_file)
                text = cleanResume(text)
                cleaned_text=preprocess(text)
               
                target = ({0:'Peoplesoft',2:'SQL Developer',1:'React JS Developer',3:'Workday'})
                
                predicted= loaded_model.predict(vector.transform([cleaned_text]))[0]
                
                string='The Uploaded Resume belongs to '+(target.get(predicted))
                st.header(string)
                
                # Extract skills
                extracted_skills = extract_skills(cleaned_text)
                if extracted_skills:
                    st.subheader('Extracted Skills:')
                    for skill in extracted_skills:
                        st.write("-", skill)
                st.header('WordCloud')
                display_wordcloud(cleaned_text)

            

if __name__ == '__main__':
    main()
