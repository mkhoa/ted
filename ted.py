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

@st.cache
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
@st.cache
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
# Images
@st.cache
def load_image(img):
  im = Image.open(os.path.join(img))
  return im

# Function to load Dataset
@st.cache(persist=True)
def load_data(query):
  df = pd.read_gbq(query, project_id='optical-scarab-285012')
  return df

# Load cover image
st.image('https://gln.edu.vn/wp-content/uploads/2019/09/5-ly-ban-nen-hoc-tieng-anh-qua-ted-talks-01.png')

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
**Schema:**
* name: The unique name of the TED Talk. Includes the title and the speaker.
* title: the title of the talk
* main_speaker: name of main speaker
* description: A summary of what the talk is about
* speaker_occupation: The occupation of the main speaker.
* num_speaker: The number of speakers in the talk.
* duration: The duration of the talk in seconds.
* event: The event where the talk took place.
* film_date: The Unix timestamp of the filming.
* published_date: The Unix timestamp for the publication of the talk on TED.com
* **comments: The number of first level comments made on the talk.**
* tags: The talk's content, identify by tags
* languages: The number of languages in which the transcript of the talk is availabel
* ratings: A stringified dictionary of the various ratings given to the talk (maybe it is being classified based on comment keywords)
* related_talks: A list of dictionaries of recommended talks to watch next.
* url: The URL of the talk.
* **views: The number of views on the talk on TED Website**
""")

#Load Data
data = load_data('SELECT * FROM `optical-scarab-285012.ted_talk.Ted_Talk`')
data_related = load_data('SELECT * FROM `optical-scarab-285012.ted_talk.Ted_Related`')
data_ratings = load_data('SELECT * FROM `optical-scarab-285012.ted_talk.Ted_Ratings`')
data_tags = load_data('SELECT * FROM `optical-scarab-285012.ted_talk.Ted_Tags`')

#convert time into date format
df = data.copy()
df_related = data_related.copy()
df_ratings = data_ratings.copy()
df_tags = data_tags.copy()
df['published_year'] = df['published_date'].apply(lambda x: datetime.datetime.utcfromtimestamp(x).strftime('%Y'))
df['published_month'] = df['published_date'].apply(lambda x: datetime.datetime.utcfromtimestamp(x).strftime('%Y%m'))
df['film_date'] = df['film_date'].apply(lambda x: datetime.datetime.utcfromtimestamp(x).strftime('%Y-%m-%d'))
df['published_date'] = df['published_date'].apply(lambda x: datetime.datetime.utcfromtimestamp(x).strftime('%Y-%m-%d'))
# Time problem. Older videos exist for a long time so many people can view it more than newer videos.
df['days_since_published'] = df['published_date'].apply(lambda x: days_between(x, '2017-09-21'))
df['view_per_day'] = df.apply(lambda x: 0 if x['days_since_published'] == 0 else round(x['views']/x['days_since_published'],0), axis = 1)
df['speaker_occupation'].fillna(value='Unidentified', inplace=True)
df['speaker_occupation_list'] = df['speaker_occupation'].apply(occpupationToList)

# Show preview of the data set
if st.checkbox("Preview Dataset"):
    if st.button("Head"):
        st.write(df.head())
    elif st.button("Tail"):
        st.write(df.tail())
    else:
        st.write(df.sample(10))

# Show entire dataset
if st.checkbox("Show All Dataset"):
    st.dataframe(df)

# Show Column Name
if st.checkbox("Show Column Names"):
    st.write(df.columns)

# Show Dimensions
df_dim = st.radio("Select Demension", ("Rows", "Columns", "All"))
if df_dim == 'Rows':
    st.text("Showing Rows")
    st.write(df.shape[0])
elif df_dim == 'Columns':
    st.text("Showing Columns")
    st.write(df.shape[1])
else:
    st.text("Showing Shape")
    st.write(df.shape)

# Show Summary
if st.checkbox("Show Summary of Dataset"):
    st.write(df.describe())

# Visualization
st.write("""
# Visualization
## Distribution and Correlation
""")
df.hist(linewidth=1.2, figsize=(12, 12))
st.pyplot()

# Show distribution plot by slider
st.write("""
Want to remove outlier?
""")
st.write("""
### Distribution of views
#### View statistics
""", df['views'].describe(), """#### Show plot""")
if st.checkbox("Show Plot"):
    sns.distplot(df[df['views'] < st.slider('Select number of views', int(df['views'].min()), int(df['views'].max()), int(df['views'].mean()))]['views'])
    st.pyplot()

st.write("""
## Correlation Heatmap
""")
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True)
st.pyplot()


# Show correlation
st.write("""
### Correlation between views and other factors
""")
factor = st.selectbox("Select factor", ("comment", "language", "duration"))
if factor == "comment":
    st.write(df[['views', 'comments']].corr())
    sns.regplot(x='comments', y='views', data=df)
    st.pyplot()
elif factor == "language":
    st.write(df[['views', 'languages']].corr())
    sns.regplot(x='languages', y='views', data=df)
    st.pyplot()
elif factor == "duration":
    st.write(df[['views', 'duration']].corr())
    sns.regplot(x='duration', y='views', data=df)
    st.pyplot()

# Talks over years
st.write("""
### Talks over years
""")
df['year'] = pd.DatetimeIndex(df['film_date']).year
year_df = pd.DataFrame(df['year'].value_counts()).reset_index()
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

st.write("""
## Bar plot of most viewed talks
""")
# Plotting
most_viewd_talks = df[['title', 'main_speaker', 'views']].sort_values('views', ascending=False)
most_viewd_talks['abrr'] = most_viewd_talks['main_speaker'].apply(lambda x: x[:3])
sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
number_of_talks = st.selectbox("Select number of talks", ("10", "20", "30"))
if number_of_talks == "10":
    chart = sns.barplot(x='main_speaker', y='views', data=most_viewd_talks[:10], ci=None)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='right')
    st.write(chart)
    st.pyplot()
elif number_of_talks == "20":
    chart = sns.barplot(x='main_speaker', y='views', data=most_viewd_talks[:20], ci=None)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='right')
    st.write(chart)
    st.pyplot()
else:
    sns.barplot(x='main_speaker', y='views', data=most_viewd_talks[:30], ci=None)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='right')
    st.write(chart)
    st.pyplot()

# Most Controversial Talk
st.write("""
## Most Controversial Talk
""")
df['controverial_rate'] = 100*df['comments']/df['views'] #comment-to-view ratio
st.write(df[['name', 'event', 'languages', 'comments', 'views', 'view_per_day', 'controverial_rate']].sort_values(by=['controverial_rate'], ascending=False).iloc[:10,:])
st.image('https://www.marketingcharts.com/wp-content/uploads/2020/04/InfluencerDB-YouTube-Influencer-Engagement-Benchmarks-Apr2020-2.png.webp')
df['controverial_category'] = df['controverial_rate'].apply(lambda x: 3 if x > 0.04 else (2 if x > 0.01 else 1))

plt.figure()
plt.pie(df.groupby('controverial_category').count().name.values,labels=["Bad", "Average", "Good"], autopct='%1.1f%%')
plt.legend(loc=4)
plt.title("TED Talk Controverial")
st.pyplot()

#Top TED Event
st.write("""
## Top TED Event
""")
st.write(df[['event', 'views']].groupby('event').sum().sort_values(['views'],ascending=False).head(10))
st.write("""
Top 10 event that has the most viewed. TED2013 and TEDGlobal 2013 combined let 2013 is a very succesful year of Ted, so higher quantity of video do not mean higher views.
""")
st.write("""
# TED Speaker
""")
#Top Speaker and their occupation
st.write(df[['main_speaker', 'speaker_occupation', 'views']].groupby(['main_speaker']).agg({'speaker_occupation': 'max', 'views': 'sum'}).sort_values(['views'],ascending=False))
#Calculate Speaker
df_speaker = df.groupby(['main_speaker']).agg({'speaker_occupation': 'max', 'name': 'count', 'views': 'sum', 'comments': 'sum'}).sort_values(['name'], ascending=False).reset_index()
df_speaker['speaker_fame'] = pd.qcut(df_speaker['views'], 3, labels=['Low', 'Average', 'High'])
df_speaker['returned_speaker'] = df_speaker['name'].apply(lambda x: True if x > 1 else False)
#Returned Speaker Percentage
plt.figure()
plt.pie(df_speaker.groupby('returned_speaker').count().name, labels=['First Time Speaker', 'Returned Speaker'], autopct='%1.1f%%')
plt.legend(loc=4)
plt.title("First Time vs Returned Speaker")
st.pyplot()

# TED Tags and Content
st.write("""
# TED Content
""")
# Top tag
st.write(df_tags.groupby('tag').count().sort_values(by=['name'], ascending=False))
# Most Freq Tags
tags_list = []
for i in df_tags['tag']:
  tags_list.append(i.lower())
createWordCloud(tags_list)
st.pyplot()

#TED Rating
rating_list = df_ratings.groupby('rating_name').sum().index.to_list()
rating_count = df_ratings.groupby('rating_name').sum()['rating_count']
plt.figure()
plt.bar(rating_list, rating_count)
plt.xlabel("Rating Name")
plt.xticks(rotation=90)
plt.ylabel("Rating Count")
plt.title("Total rating of TED Talks")
plt.ticklabel_format(style='plain', axis='y')
st.pyplot()
st.write('''Inspiring, Informative, and Fascinating are the top attribute that users are expecting from TED Talk''')


#TED Metric
st.write("""
## Metric for TED Talk
Focused on 3 metric:
- Volume: the total views of video
- Growth: Views per Days of video
- Controversial: comments-to-views ratio
""")

df['views_category'] = pd.qcut(df['views'], 3, labels=[1, 2, 3])
df['growth_category'] = pd.qcut(df['view_per_day'], 3, labels=[1, 2, 3])
df['score'] = df['views_category'].astype(int) + df['growth_category'].astype(int) + df['controverial_category'].astype(int)
df['score_group'] = pd.Categorical(df['score'].apply(lambda x: 'Good' if x > 6 else ('Average' if x > 4 else 'Bad')), ["Bad", "Average", "Good"])
st.write(df.sample(5).astype('object'))

# Return Speaker
df = df.merge(df_speaker, on=['main_speaker'], how='left', suffixes=(None,"_y"))
rspk_analyst = df.groupby('score_group').sum()['returned_speaker']/df.groupby('score_group').count()['returned_speaker']
plt.figure()
plt.bar(rspk_analyst.index, rspk_analyst.values)
plt.xlabel("Score Group")
plt.ylabel("Percentage of Returned Speaker")
st.pyplot()
st.write("""Good videos have the highest percentage of returned speaker (30%), but the difference is not huge between three group.""")

a = df[df['returned_speaker'] == 1]['score'] # Speak with returned speaker
b = df[df['returned_speaker'] == 0]['score'] # Speak with first time speaker
st.write(ttest_ind(a, b, equal_var=False))

st.write('''
**Student t-test**
This is a two-sided test for the null hypothesis that 2 independent samples have identical average (expected) values. This test assumes that the populations have identical variances by default.

When apply t-test on two sample "Speak with returned speaker" and "Speak with first time speaker", the p-value is 0.13, which is not a good number for scientific research, but in this we can still conclude that the  difference  is statistic significant, returned speaker can have a positive impact of TED Talks quality.''')

# Most Freq Occupation keyword in Good Ted Talk
st.write("""
### Most Freq Occupation keyword in Good Ted Talk
""")
tags_list = []
for i in df[df['score_group']=='Good']['speaker_occupation']:
  tags_list.append(i.lower())
createWordCloud(tags_list)
st.pyplot()

st.write('''
**TEDx**

TEDx is a TED program that enables volunteers to independently organize TED-like events in their local communities. Under a license granted by TED, hardworking volunteers produce events and curate speakers to celebrate local ideas, shift perspectives, and ultimately change lives. see less

**TEDFellows**

The TED Fellows program hand-picks young innovators from around the world to raise international awareness of their work and maximize their impact.'''
)

df['tedx'] = df['tags'].apply(lambda x: True if 'TEDx' in x else False)
df['tedfellows'] = df['tags'].apply(lambda x: True if 'ellows' in x else False)
tedx_analyst = df.groupby('score_group').sum()['tedx']/df.groupby('score_group').count()['tedx']
plt.figure()
plt.bar(tedx_analyst.index, tedx_analyst.values)
plt.xlabel("Score Group")
plt.ylabel("Percentage of Tedx Speaker")
st.pyplot()

a = df[df['tedx'] == 1]['score'] # Speak with returned speaker
b = df[df['tedx'] == 0]['score'] # Speak with first time speaker
st.write(ttest_ind(a, b, equal_var=False))

tedfellow_analyst = df.groupby('score_group').sum()['tedfellows']/df.groupby('score_group').count()['tedfellows']
plt.figure()
plt.bar(tedfellow_analyst.index, tedfellow_analyst.values)
plt.xlabel("Score Group")
plt.ylabel("Percentage of Fellow Speaker")
st.pyplot()
a = df[df['tedfellows'] == 1]['score'] # Speak with returned speaker
b = df[df['tedfellows'] == 0]['score'] # Speak with first time speaker
st.write(ttest_ind(a, b, equal_var=False))

st.write('''
**Conclusion**
* Returned Speaker can have positive impact on TED Talk video
* User are expecting Inspiring, Informative, and Fascinating from TED Videos
* Having translated transcript can increase the reach of TED videos and get more views
* Hot trend right now for TED video is: Technology, Science and Global Issues''')