import datetime
import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import os
import warnings                 
import json
import re
import ast
import collections
import wordcloud
from google.colab import auth
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