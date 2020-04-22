#!/usr/bin/env python
# coding: utf-8

# In[1]:


import boto3
import pandas as pd

s3 = boto3.resource('s3')
# bucket = s3.Bucket('grp5')
# Iterates through all the objects, doing the pagination for you. Each obj
# is an ObjectSummary, so it doesn't contain the body. You'll need to call
# get to get the whole body.

for bucket in s3.buckets.all():
    for obj in bucket.objects.all():
#         key = obj.key
#         body = obj.get()['Body'].read()
        print(obj.key)
        print(bucket.name)


# In[7]:


import os
import pandas as pd

bucket_name = 'mentalhealthsurveydatabucket'
object_key = 'Data/survey.csv'

path = 's3://{}/{}'.format(bucket_name, object_key)

print(path)
df = pd.read_csv(path)


# In[9]:


df.head()


# In[ ]:




