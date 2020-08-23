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
from google.oauth2 import service_account
from scipy.stats import ttest_ind, ttest_ind_from_stats
from scipy.special import stdtr
from PIL import Image

##Google
credentials = service_account.Credentials.from_service_account_file('/home/mkhoa/Credentials/optical-scarab-285012-082a9dd1bfc1.json')

pd.read_gbq('SELECT * FROM `optical-scarab-285012.ted_talk.Ted_Talk`', project_id='optical-scarab-285012', credentials=credentials, use_bqstorage_api=True)
help(pd.read_gbq)