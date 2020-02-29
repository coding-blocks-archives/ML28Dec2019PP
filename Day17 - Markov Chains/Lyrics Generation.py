#!/usr/bin/env python
# coding: utf-8

# In[ ]:


text = "the man was .... they .... then . ... the ,,,, they ... the "

X   ,   y   - freq
the    "_"  -  3 
the    "y" -  2
the    "n"  - 1


# In[46]:


import numpy as np


# In[11]:


def generateTable(data, k = 4):
    T = {}
    for i in range(len(data) - k):
        X = data[i:i+k]
        y = data[i+k]
        
        if T.get(X) is None:
            T[X] = {}
            T[X][y] = 1
        else:
            if T[X].get(y) is None:
                T[X][y] = 1
            else:
                T[X][y] +=1

    return T


# In[16]:


data = "dog is bitting hello hello helli dog is running"


# In[18]:


T = generateTable(data)


# In[37]:


def convertfreqIntoProb(T):

    for kx in T.keys():
        s = sum(list(T[kx].values()))
        
        for k in T[kx].keys():
            T[kx][k] = T[kx][k]/s
            
    return T


# In[39]:


T = convertfreqIntoProb(T)


# In[56]:


with open("english_speech_2.txt") as f:
    data = f.read().lower()


# In[57]:


print(data)


# In[58]:


T = generateTable(data)
T = convertfreqIntoProb(T)


# In[60]:


len(T)


# ## Sampling

# In[51]:


l = ["apple", "mango", "banana", "orange"]
probabs = [0.5, 0.3, 0.15, 0.05]


# In[55]:


for i in range(20):
    print(np.random.choice(l, p=probabs) )


# ## Generate Text

# In[76]:


def sample_next(ctx, T, k = 4):
    ctx = ctx[-k:]
    
    if T.get(ctx) is None:
        return " "
    possible_chars = list(T[ctx].keys())
    possible_porabs = list(T[ctx].values())
    
    return np.random.choice(possible_chars, p = possible_porabs)


# In[86]:


sample_next("the ", T)


# In[92]:


def generateText(starting_sentence, T, k = 4, max_len = 2000):
    sentence = starting_sentence

    
    for ix in range(max_len):
        
        next_char = sample_next(sentence, T, k )
        sentence += next_char
        
    return sentence


# In[94]:


print(generateText("dear", T))


# In[ ]:




