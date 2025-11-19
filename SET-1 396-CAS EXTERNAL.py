#!/usr/bin/env python
# coding: utf-8

# In[1]:


#SET-1
def find_common_elements(list1, list2):
    common = []
    for item in list1:
        if item in list2 and item not in common:
            common.append(item)
    return common
# Example
list1 = [1, 2, 3, 4]
list2 = [3, 4, 5, 6]
result = find_common_elements(list1, list2)
print(result)


# In[3]:


#SET-1
def word_frequency(sentence):
    sentence = sentence.lower()
    words = sentence.split()
    freq = {}
    for word in words:
        if word not in freq:
            freq[word] = 1
        else:
            freq[word] += 1
    return freq
# Example
sentence = "the cat and the hat"
result = word_frequency(sentence)
print(result) 


# In[4]:


#SET-3
[[1, 2], [3, 4], [5, 6]]


# In[ ]:




