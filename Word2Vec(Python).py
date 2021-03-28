#!/usr/bin/env python
# coding: utf-8

# # CS381 NLP HW#2
# # Word2Vec

# ## Imports and logging
# 
# First, we start with our imports and get logging established:

# In[1]:


# imports needed and set up logging
# import gzip
import gensim 
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# <h2>Reading a input file </h2>

# In[2]:


data_file="news.crawl"

with open ('news.crawl', 'rb') as f:
    for i,line in enumerate (f):
        print(line)
        break


# In[3]:



def read_input(input_file):
    """This method reads the input file which is in gzip format"""
    
    logging.info("reading file {0}...this may take a while".format(input_file))
    
    with open (input_file, 'rb') as f:
        for i, line in enumerate (f): 

            if (i%10000==0):
                logging.info ("read {0} reviews".format (i))
            # do some pre-processing and return a list of words for each review text
            yield gensim.utils.simple_preprocess (line)

# read the tokenized reviews into a list
# each review item becomes a serries of words
# so this becomes a list of lists
documents = list (read_input (data_file))
logging.info ("Done reading data file")    


# ## Training the Word2Vec model

# In[4]:


model1 = gensim.models.Word2Vec (documents, size=150, window=5, min_count=2, workers=10, iter=10)
model1.train(documents,total_examples=len(documents),epochs=10)


# ## Quesitons:

# ### Q1 Report similarity scores for the following pairs: (dirty, clean), (big, dirty),(big, large) , (big,small)

# In[5]:


# similarity score between two words, "dirty" and "clean".
model1.wv.similarity(w1="dirty",w2="clean")


# In[6]:


# similarity score between two words, "big" and "dirty"
model1.wv.similarity(w1="big",w2="dirty")


# In[7]:


# similarity score between two words, "big" and "large"
model1.wv.similarity(w1="big",w2="large")


# In[8]:


# similarity score between two words, "big" and "small"
model1.wv.similarity(w1="big",w2="small")


# ### Q2 Report 5 most similar items and the scores to ’polite’, ’orange’

# In[9]:


#* look up top 5 words similar to 'polite'
w1 = ["polite"]
model1.wv.most_similar (positive=w1,topn=5)


# In[10]:


#* look up top 5 words similar to 'orange'
w1 = ["orange"]
model1.wv.most_similar (positive=w1,topn=5)


# ### Q3 A Model with other parameters: size =50, window = 2 

# In[11]:


model2 = gensim.models.Word2Vec (documents, size=50, window=2, min_count=2, workers=10, iter=10)
model2.train(documents,total_examples=len(documents),epochs=10)


# ### Q3-1 Report similarity scores for the following pairs: (dirty, clean), (big, dirty),(big, large) , (big,small)

# In[12]:


# similarity score between two words, "dirty" and "clean"
model2.wv.similarity(w1="dirty",w2="clean")


# In[13]:


# similarity score between two words, "big" and "dirty"
model2.wv.similarity(w1="big",w2="dirty")


# In[14]:


# similarity score between two words, "big" and "large"
model2.wv.similarity(w1="big",w2="large")


# In[15]:


# similarity score between two words, "big" and "small"
model2.wv.similarity(w1="big",w2="small")


# ### Q3-2 Report 5 most similar items and the scores to ’polite’, ’orange’

# In[16]:


# look up top 5 most similar words to 'polite'
w1 = ["polite"]
model2.wv.most_similar (positive=w1,topn=5)


# In[17]:


# look up top 5 most similar words to 'orange'
w1 = ["orange"]
model2.wv.most_similar (positive=w1,topn=5)


# In[ ]:




