import datetime
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import os            
import json
import re
import ast
import collections
import wordcloud
from google.colab import auth
from google.colab import drive
from scipy.stats import ttest_ind, ttest_ind_from_stats
from scipy.special import stdtr
from PIL import Image

## Cleaning Functions
'''This Section contain needed custom functions

'''
def outlinerDelete (dataframe):
  Q1 = dataframe.quantile(0.25)
  Q3 = dataframe.quantile(0.75)
  IQR = Q3 - Q1
  return dataframe[(df_normalize > (Q1 - 1.5 * IQR)) & (df_normalize < (Q3 + 1.5 * IQR))]

def stringToList (data):
  words = [';',',','and ','+']
  for word in words:
    #print (data.split(word))
    result = [x.strip() for x in data.split(word)]
    if len(result) > 1:
      break
  return result

def occpupationToList (data):
  words = [';',',','and ','+']
  for word in words:
    #print (data.split(word))
    result = [x.strip() for x in data.split(word)]
    if len(result) > 1:
      break
  return result

# Pie Chart of Speaker Occupation
def extractElement_fromlist (serie):
  result = []
  for i in serie:
    result.extend(i)
    #print (i)
  return result  

#Date Difference
def days_between(d1, d2):
    d1 = datetime.datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)

#Create World Cloud
def createWordCloud (text):
  occurrences = collections.Counter(text)
  cloud = wordcloud.WordCloud(background_color="white", width=1920, height=1080, min_font_size=8)
  cloud.generate_from_frequencies(occurrences)
  myimage = cloud.to_array()
  plt.imshow(myimage, interpolation = 'nearest')
  plt.axis('off')
  plt.show()


## Streamlit
'''Initiate Streamlit

'''
# Images
@st.cache
def load_image(img):
    im = Image.open(os.path.join(img))
    return im

# Function to load Dataset
@st.cache(persist=True)
def explore_data(dataset):
    df = pd.read_csv(os.path.join(dataset))
    return df

# Load cover image
st.image(load_image('/content/TED.jpg'))

# Title
st.title("TED EDA App")
st.text("A Ted talk about Ted talks")

# Ted Description
st.write("""
# What is TED?
TED is a nonprofit devoted to spreading ideas,
usually in the form of short, powerful talks (18 minutes or less).
TED began in 1984 as a conference where Technology,
Entertainment and Design converged, and today covers almost
all topics — from science to business to global issues — in more than 100 languages.
Meanwhile, independently run TEDx events help share ideas in communities around the world.""")

# Data set description
st.write("""
# About the Dataset
These datasets contain information about all audio-video recordings of
TED Talks uploaded to the official TED.com website until September 21st, 2017.
The TED main dataset contains information about all talks
including number of views, number of comments, descriptions, speakers and titles.
The TED transcripts dataset contains the transcripts for all talks available on TED.com.""")

# Acknowledgements
st.write("""
# Acknowledgements
The data has been scraped from the official TED Website by Rounal Banik""")

# Question we want to answer
st.write("""
# Question we want to answer
    1. Which speaker should we invite?
    2. What video categories and contents should we focus?
    3. What are our users expect from video?""")

# Overview of the data set
st.write("""
# TED EDA
## Overview of the dataset

""")

# Load dataset
df = pd.read_gbq('SELECT * FROM `optical-scarab-285012.ted_talk.Ted_Talk`', project_id='optical-scarab-285012')
data = explore_data(df)

# Reorder the columns
data = data[['title', 'main_speaker', 'views', 'published_date', 'comments', 'description', 'name', 'speaker_occupation', 'num_speaker', 'duration', 'event', 'film_date' , 'tags', 'languages', 'ratings', 'related_talks', 'url']]

# Show preview of the data set
if st.checkbox("Preview Dataset"):
    if st.button("Head"):
        st.write(data.head())
    elif st.button("Tail"):
        st.write(data.tail())
    else:
        st.write(data.sample(10))

# Show entire dataset
if st.checkbox("Show All Dataset"):
    st.dataframe(data)

# Show Column Name
if st.checkbox("Show Column Names"):
    st.write(data.columns)

# Show Dimensions
data_dim = st.radio("Select Demension", ("Rows", "Columns", "All"))
if data_dim == 'Rows':
    st.text("Showing Rows")
    st.write(data.shape[0])
elif data_dim == 'Columns':
    st.text("Showing Columns")
    st.write(data.shape[1])
else:
    st.text("Showing Shape")
    st.write(data.shape)

# Show Summary
if st.checkbox("Show Summary of Dataset"):
    st.write(data.describe())

# Select A Columns
col_option = st.selectbox("Select Column", ("views", "duration", "film_date",
                                            "languages", "comments"))
if col_option == "views":
    st.write(data['views'])
elif col_option == "comments":
    st.write(data['comments'])


# Visualization
st.write("""
## Visualization
### Bar plot of most viewed talks
""")
# Plotting
most_viewd_talks = data[['title', 'main_speaker', 'views']].sort_values('views', ascending=False)
most_viewd_talks['abrr'] = most_viewd_talks['main_speaker'].apply(lambda x: x[:3])
sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
number_of_talks = st.selectbox("Select number of talks", ("10", "20", "30"))
if number_of_talks == "10":
    st.write(sns.barplot(x='abrr', y='views', data=most_viewd_talks[:10], ci=None))
    st.pyplot()
elif number_of_talks == "20":
    st.write(sns.barplot(x='abrr', y='views', data=most_viewd_talks[:20], ci=None))
    st.pyplot()
else:
    st.write(sns.barplot(x='abrr', y='views', data=most_viewd_talks[:30], ci=None))
    st.pyplot()

# Show distribution plot
st.write("""
### Distribution of views
#### View statistics
""", data['views'].describe(), """#### Show plot""")

if st.checkbox("Show Plot"):
    sns.distplot(data[data['views'] < st.slider('Select number of views', int(data['views'].min()), int(data['views'].max()), int(data['views'].mean()))]['views'])
    st.pyplot()

# Show correlation
st.write("""
### Correlation between views and other factors
""")
factor = st.selectbox("Select factor", ("comment", "language", "duration"))
if factor == "comment":
    st.write(data[['views', 'comments']].corr())
    sns.regplot(x='comments', y='views', data=data)
    st.pyplot()
elif factor == "language":
    st.write(data[['views', 'languages']].corr())
    sns.regplot(x='languages', y='views', data=data)
    st.pyplot()
elif factor == "duration":
    st.write(data[['views', 'duration']].corr())
    sns.regplot(x='duration', y='views', data=data)
    st.pyplot()

# Talks over years
st.write("""
### Talks over years
""")

data['year'] = pd.DatetimeIndex(data['film_date']).year
year_df = pd.DataFrame(data['year'].value_counts()).reset_index()
year_df.columns = ['year', 'count']
year_df = year_df.sort_values('year')

film_year = st.selectbox("Select factor", ("From 1972", "From 2006"))
if film_year == "From 1972":
    plt.figure(figsize=(18, 5))
    sns.pointplot(x='year', y='count', data=year_df)
    st.pyplot()
elif film_year == "From 2006":
    plt.figure(figsize=(18, 5))
    sns.pointplot(x='year', y='count', data=year_df[12:])
    st.pyplot()