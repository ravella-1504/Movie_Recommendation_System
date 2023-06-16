#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


# loading the data from the csv file to apandas dataframe
movies_data = pd.read_csv("C:\\Users\\lavan\\Downloads\\movies.csv")


# In[3]:


# printing the first 5 rows of the dataframe
movies_data.head()


# In[4]:


# number of rows and columns in the data frame

movies_data.shape


# In[5]:


# selecting the relevant features for recommendation

selected_features = ['genres','keywords','tagline','cast','director']
print(selected_features)


# In[6]:


# replacing the null valuess with null string

for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')


# In[7]:


# combining all the 5 selected features

combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']


# In[8]:


print(combined_features)


# In[9]:


# converting the text data to feature vectors
vectorizer = TfidfVectorizer()


# In[11]:


feature_vectors = vectorizer.fit_transform(combined_features)
print(feature_vectors)


# In[13]:


# getting the similarity scores using cosine similarity

similarity = cosine_similarity(feature_vectors)
print(similarity)


# In[14]:


print(similarity.shape)


# In[23]:


# getting the movie name from the user

movie_name = input(' Enter your favourite movie name : ')


# In[24]:


# creating a list with all the movie names given in the dataset

list_of_all_titles = movies_data['title'].tolist()
print(list_of_all_titles)


# In[25]:


# finding the close match for the movie name given by the user

find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
print(find_close_match)


# In[26]:


close_match = find_close_match[0]
print(close_match)


# In[27]:


# finding the index of the movie with title

index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
print(index_of_the_movie)


# In[28]:


# getting a list of similar movies

similarity_score = list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)


# In[29]:


len(similarity_score)


# In[ ]:




