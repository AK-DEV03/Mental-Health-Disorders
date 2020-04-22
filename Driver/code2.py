#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement:
# 
# ### To identify the strongest predictors of mental health illness & to predict if a person needs mental treatment in the workplace using SageMaker and other AWS services.

# In[544]:


import warnings
warnings.filterwarnings('ignore')


# In[545]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Set matplotlib sizes
plt.rc('font', size=20)
plt.rc('axes', titlesize=20)
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rc('legend', fontsize=20)
plt.rc('figure', titlesize=20)


# In[637]:


import boto3
import sagemaker
from sagemaker import get_execution_role
s3 = boto3.resource('s3')
# bucket = s3.Bucket('grp5')
# Iterates through all the objects, doing the pagination for you. Each obj
# is an ObjectSummary, so it doesn't contain the body. You'll need to call
# get to get the whole body.

for bucket in s3.buckets.all():
    for obj in bucket.objects.all():
#         key = obj.key
#         body = obj.get()['Body'].read()
        print(bucket.name)
        print(obj.key)


# In[547]:


import os

bucket_name = 'mentalhealthsurveydatabucket'
object_key = 'Data/survey.csv'

path = 's3://{}/{}'.format(bucket_name, object_key)

print(path)


# In[548]:


## Loading data
import pandas as pd

# Load the raw data
df_raw = pd.read_csv(path)
df_raw


# In[549]:


##Divide the data into train and test
from sklearn.model_selection import train_test_split

# Divide the training data into training (80%) and test (20%)
df_raw_train, df_raw_test = train_test_split(df_raw, train_size=0.8, random_state=42)

# Reset the index
df_raw_train, df_raw_test = df_raw_train.reset_index(drop=True), df_raw_test.reset_index(drop=True)


# In[550]:


# Make a copy of df_raw_train
df_train = df_raw_train.copy(deep=True)

# Make a copy of df_raw_test
df_test = df_raw_test.copy(deep=True)

# Get the name of the target
target = 'treatment'


# In[551]:


# Print the dimension of df_train
pd.DataFrame([[df_train.shape[0], df_train.shape[1]]], columns=['# rows', '# columns'])


# In[552]:


# Print the dimension of df_test
pd.DataFrame([[df_test.shape[0], df_test.shape[1]]], columns=['# rows', '# columns'])


# In[553]:


# Print the first 10 rows of df_train
df_train.head(10)


# In[554]:


#Dividing the Training data into Training and Validation
# Divide the training data into training (80%) and validation (20%)
df_train, df_valid = train_test_split(df_train, train_size=0.8, random_state=42)

# Reset the index
df_train, df_valid = df_train.reset_index(drop=True), df_valid.reset_index(drop=True)


# In[555]:


#Handling the Identifiers

# Combine df_train, df_valid and df_test
df = pd.concat([df_train, df_valid, df_test], sort=False)

df.dtypes


# ### Identifying the identifiers
# The code below shows how to find identifiers from the data.

# In[556]:


def id_checker(df):
    """
    The identifier checker

    Parameters
    ----------
    df : dataframe
    
    Returns
    ----------
    The dataframe of identifiers
    """
    
    # Get the identifiers
    df_id = df[[var for var in df.columns 
                if (df[var].dtype != 'float'
                    and df[var].nunique(dropna=True) == df[var].notnull().sum())]]
    
    return df_id


# In[557]:


# Call id_checker on df
df_id = id_checker(df)

# Print the first 5 rows of df_id
df_id.head()


# ### Removing the Identifiers
# The code below shows how to remove the identifiers from data (using pandas DataFrame.drop).

# In[558]:


import numpy as np

# Remove the identifiers from df_train
df_train = df_train.drop(columns=np.intersect1d(df_id.columns, df_train.columns))

# Remove the identifiers from df_valid
df_valid = df_valid.drop(columns=np.intersect1d(df_id.columns, df_valid.columns))

# Remove the identifiers from df_test
df_test = df_test.drop(columns=np.intersect1d(df_id.columns, df_test.columns))


# ### Combining the training, validation and testing data
# The code below shows how to combine the training, validation and testing data (using pandas concat).

# In[559]:


# Combine df_train, df_valid and df_test
df = pd.concat([df_train, df_valid, df_test], sort=False)


# ### Identifying missing values
# The code below shows how to find variables with NaN (using pandas DataFrame.isna), their proportion of NaN and 
# dtype (data type objects, using pandas Series.dtype).

# In[560]:


def nan_checker(df):
    """
    The NaN checker

    Parameters
    ----------
    df : dataframe
    
    Returns
    ----------
    The dataframe of variables with NaN, their proportion of NaN and dtype
    """
    
    # Get the dataframe of variables with NaN, their proportion of NaN and dtype
    df_nan = pd.DataFrame([[var, df[var].isna().sum() / df.shape[0], df[var].dtype]
                           for var in df.columns if df[var].isna().sum() > 0],
                          columns=['var', 'proportion', 'dtype'])
    
    # Sort df_nan in accending order of the proportion of NaN
    df_nan = df_nan.sort_values(by='proportion', ascending=False).reset_index(drop=True)
    
    return df_nan


# In[561]:


# Call nan_checker on df
df_nan = nan_checker(df)

# Print df_nan
df_nan


# In[562]:


# Print the unique dtype of the variables with NaN
pd.DataFrame(df_nan['dtype'].unique(), columns=['dtype'])


# In[563]:


import numpy as np
# Print df_miss
df_miss = ['Timestamp', 'comments', 'state']

df_train = df_train.drop(df_miss, axis= 1)
df_valid = df_valid.drop(df_miss, axis= 1)
df_test = df_test.drop(df_miss, axis= 1)


# In[564]:


# Print the dimension of df_remove_train
pd.DataFrame([[df_train.shape[0], df_train.shape[1]]], columns=['# rows', '# columns'])


# In[565]:


# Print the dimension of df_remove_valid
pd.DataFrame([[df_valid.shape[0], df_valid.shape[1]]], columns=['# rows', '# columns'])


# In[566]:


# Print the dimension of df_remove_test
pd.DataFrame([[df_test.shape[0], df_test.shape[1]]], columns=['# rows', '# columns'])


# In[567]:


# Combine df_train, df_valid and df_test
df = pd.concat([df_train, df_valid, df_test], sort=False)
pd.DataFrame(df['self_employed'].unique(), columns=['self_employed'])
df['self_employed'] = df['self_employed'].replace(np.nan,'No')
df['self_employed'].unique()


# In[568]:


#Removes Age values <18 and >100
df.drop( df[ ( df['Age'] < 18 ) ^ (df['Age'] > 100) ].index , inplace=True)
pd.DataFrame([[df.shape[0], df.shape[1]]], columns=['# rows', '# columns'])


# In[569]:


pd.DataFrame(df['Gender'].unique(), columns=['Gender'])


# In[570]:


#clean 'Gender'
#Slower case all columm's elements
gender = df['Gender'].str.lower()
#print(gender)

#Select unique elements
gender = df['Gender'].unique()

#Made gender groups
male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]
trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]           
female_str = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]


for (row, col) in df.iterrows():

    if str.lower(col.Gender) in male_str:
        df['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)

    if str.lower(col.Gender) in female_str:
        df['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)

    if str.lower(col.Gender) in trans_str:
        df['Gender'].replace(to_replace=col.Gender, value='trans', inplace=True)


# In[571]:


pd.DataFrame(df['Gender'].unique(), columns=['Gender'])


# In[572]:


newDF = {}

for column in df.columns[0:]:
    if column != 'Age' and column != 'Country':
          data = list(df[column].unique())
          newDF[column] = data
        
newDF


# ### Identifying the Categorical Variables
# The code below shows how to find the categorical variables that have object as dtype (using pandas.Series.dtype).

# In[573]:


def cat_var_checker(df):
    """
    The categorical variable checker

    Parameters
    ----------
    df: the dataframe
    
    Returns
    ----------
    The dataframe of categorical variables and their number of unique value
    """
    
    # Get the dataframe of categorical variables and their number of unique value
    df_cat = pd.DataFrame([[var, df[var].nunique(dropna=False)]
                           for var in df.columns if df[var].dtype == 'object'],
                          columns=['var', 'nunique'])
    
    # Sort df_cat in accending order of the number of unique value
    df_cat = df_cat.sort_values(by='nunique', ascending=False).reset_index(drop=True)
    
    return df_cat


# In[574]:


# Call cat_var_checker on df
df_cat = cat_var_checker(df)

# Print the dataframe
df_cat


# ### Encoding the categorical features
# The code below shows how to encode the categorical features in the combined data (using pandas.get\_dummies).

# In[575]:


# One-hot-encode the categorical features in the combined data
df = pd.get_dummies(df, columns=np.setdiff1d(df_cat['var'], [target]))

# Print the first 10 rows (samples) of df
df.head(10)


# ### Encoding the categorical target
# The code below shows how to encode the categorical target in the combined data (using sklearn.LabelEncoder).

# In[576]:


from sklearn.preprocessing import LabelEncoder

# The LabelEncoder
le = LabelEncoder()

# Encode the categorical target in the combined data
df[target] = le.fit_transform(df[target].astype(str))

# Print the first 5 rows of df
df.head()


# In[577]:


##Divide the data into train and test
from sklearn.model_selection import train_test_split

# Divide the training data into training (80%) and test (20%)
df_train, df_test = train_test_split(df, train_size=0.8, random_state=42, stratify = df[target])

# Reset the index
df_train, df_test = df_train.reset_index(drop=True), df_test.reset_index(drop=True)

# Divide the training data into training (80%) and test (20%)
df_train, df_valid = train_test_split(df_train, train_size=0.8, random_state=42)

# Reset the index
df_train, df_valid = df_train.reset_index(drop=True), df_valid.reset_index(drop=True)


# In[578]:


# Print the dimension of df_train
pd.DataFrame([[df_train.shape[0], df_train.shape[1]]], columns=['# rows', '# columns'])


# In[579]:


# Print the dimension of df_valid
pd.DataFrame([[df_valid.shape[0], df_valid.shape[1]]], columns=['# rows', '# columns'])


# In[580]:


# Print the dimension of df_test
pd.DataFrame([[df_test.shape[0], df_test.shape[1]]], columns=['# rows', '# columns'])


# ## Getting the name of the features

# In[581]:


features = np.setdiff1d(df.columns, [target])


# In[582]:


index_list = list(df_train.index)

np.random.seed(5)
np.random.shuffle(index_list)

print(index_list)


# In[583]:


df_train = df_train.iloc[index_list,:]


# ## Getting the feature matrix and target vector

# In[633]:


# Get the feature matrix
X_train = df_train[features].to_numpy()
X_valid = df_valid[features].to_numpy()
X_test = df_test[features].to_numpy()

# Get the target vector
y_train = df_train[target].astype(int).to_numpy()
y_valid = df_valid[target].astype(int).to_numpy()
y_test = df_test[target].astype(int).to_numpy()
y_test1 = df_test[target].astype(int).to_numpy()


# ## Scaling the data
# The code below shows how to standardize the data (using sklearn StandardScaler).

# In[585]:


from sklearn.preprocessing import StandardScaler

# The StandardScaler
ss = StandardScaler()

# Standardize the training data
X_train = ss.fit_transform(X_train)

# Standardize the validation data
X_valid = ss.transform(X_valid)

# Standardize the testing data
X_test = ss.transform(X_test)


# In[ ]:





# In[586]:


newDF = {}

for column in df.columns[0:]:
    if column != 'Age' and column != 'Country':
          data = list(df[column].unique())
          newDF[column] = data
        
newDF


# # Hyperparameter Tuning and Model Selection

# ## Creating the dictionary of the models
# - In the dictionary:
#     - the key is the acronym of the model
#     - the value is the model

# In[587]:


get_ipython().system('pip install -U xgboost')


# In[588]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.experimental import enable_hist_gradient_boosting
# from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.neural_network import MLPClassifier
# import sagemaker.xgboost as XGBClassifier

models = {'lr': LogisticRegression(class_weight='balanced', random_state=42),
          'dtc': DecisionTreeClassifier(class_weight='balanced', random_state=42),
          'rfc': RandomForestClassifier(class_weight='balanced', random_state=42),
#           'hgbc': HistGradientBoostingClassifier(random_state=42),
          'xgbc': XGBClassifier(seed=42),
          'mlpc': MLPClassifier(early_stopping=True, random_state=42)}


# ## Creating the dictionary of the pipelines
# In the dictionary:
# - the key is the acronym of the model
# - the value is the pipeline, which, for now, only includes the model

# In[589]:


from sklearn.pipeline import Pipeline

pipes = {}

for acronym, model in models.items():
    pipes[acronym] = Pipeline([('model', model)])


# ## Hyperparameter tuning and  model selection using Sklearn (0.22 or above)

# ### Getting the predefined split cross-validator

# In[590]:


from sklearn.model_selection import PredefinedSplit

# Combine the feature matrix in the training and validation data
X_train_valid = np.vstack((X_train, X_valid))

# Combine the target vector in the training and validation data
y_train_valid = np.append(y_train, y_valid)

# Get the indices of training and validation data
train_valid_idxs = np.append(np.full(X_train.shape[0], -1), np.full(X_valid.shape[0], 0))

# The PredefinedSplit
ps = PredefinedSplit(train_valid_idxs)


# ### Creating the dictionary of the parameter grids
# - In the dictionary:
#     - the key is the acronym of the model
#     - the value is the parameter grid of the model

# In[591]:


param_grids = {}


# #### The parameter grid for logistic regression
# The hyperparameters we want to fine-tune are:
# - C
# - tol
# 
# See details of the meaning of the hyperparametes in [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

# In[592]:


# The grids for C
C_grids = [10 ** i for i in range(-2, 3)]

# The grids for tol
tol_grids = [10 ** i for i in range(-6, -1)]

# Update param_grids
param_grids['lr'] = [{'model__C': C_grids,
                      'model__tol': tol_grids}]


# #### The parameter grid for decision tree
# The hyperparameters we want to fine-tune are:
# - min_samples_split
# - min_samples_leaf
# - max_depth
# 
# See details of the meaning of the hyperparametes in [sklearn.tree.DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)

# In[593]:


# The grids for min_samples_split
min_samples_split_grids = [2, 30, 100]

# The grids for min_samples_leaf
min_samples_leaf_grids = [1, 30, 100]

# The grids for max_depth
max_depth_grids = range(1, 11)

# Update param_grids
param_grids['dtc'] = [{'model__min_samples_split': min_samples_split_grids,
                       'model__min_samples_leaf': min_samples_leaf_grids,
                       'model__max_depth': max_depth_grids}]


# #### The parameter grid for random forest
# The hyperparameters we want to fine-tune are:
# - min_samples_split
# - min_samples_leaf
# 
# See details of the meaning of the hyperparametes in [sklearn.ensemble.RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

# In[594]:


# The grids for min_samples_split
min_samples_split_grids = [2, 20, 100]

# The grids for min_samples_leaf
min_samples_leaf_grids = [1, 20, 100]

# Update param_grids
param_grids['rfc'] = [{'model__min_samples_split': min_samples_split_grids,
                       'model__min_samples_leaf': min_samples_leaf_grids}]


# #### The parameter grid for histogram-based gradient boosting
# The hyperparameters we want to fine-tune are:
# - learning_rate
# - min_samples_leaf
# 
# See details of the meaning of the hyperparametes in [sklearn.ensemble.HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)

# In[595]:


# # The grids for learning_rate
# learning_rate_grids = [10 ** i for i in range(-3, 2)]

# # The grids for min_samples_leaf
# min_samples_leaf_grids = [1, 20, 100]

# # Update param_grids
# param_grids['hgbc'] = [{'model__learning_rate': learning_rate_grids,
#                         'model__min_samples_leaf': min_samples_leaf_grids}]


# #### The parameter grid for xgboost
# The hyperparameters we want to fine-tune are:
# - eta
# - gamma
# - lambda
# 
# See details of the meaning of the hyperparametes in [XGBoost Parameters](https://xgboost.readthedocs.io/en/latest/parameter.html)

# In[596]:


# The grids for eta
eta_grids = [10 ** i for i in range(-4, 1)]

# The grids for gamma
gamma_grids = [0, 10, 100]

# The grids for lambda
lambda_grids = [10 ** i for i in range(-4, 5)]

# Update param_grids
param_grids['xgbc'] = [{'model__eta': eta_grids,
                        'model__gamma': gamma_grids,
                        'model__lambda': lambda_grids}]


# #### The parameter grid for multi-layer perceptron classifier
# The hyperparameters we want to fine-tune are:
# - alpha
# - learning_rate_init
# 
# See details of the meaning of the hyperparametes in [sklearn.neural_network.MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)

# In[597]:


# The grids for alpha
alpha_grids = [10 ** i for i in range(-6, -1)]

# The grids for learning_rate_init
learning_rate_init_grids = [10 ** i for i in range(-5, 0)]

# Update param_grids
param_grids['mlpc'] = [{'model__alpha': alpha_grids,
                        'model__learning_rate_init': learning_rate_init_grids}]


# ### Creating the directory for the cv results.

# In[598]:


import os

# Make directory
directory = os.path.dirname('cv_results/')
if not os.path.exists(directory):
    os.makedirs(directory)


# ### Hyperparameter Tuning
# The code below shows how to fine-tune the hyperparameters of the models above (using sklearn GridSearchCV).

# In[599]:


# import scipy
# scipy.test()


# In[600]:


from sklearn.model_selection import GridSearchCV

# The list of [best_score_, best_params_, best_estimator_] obtained by GridSearchCV
best_score_param_estimator_gs = []

for acronym in pipes.keys():
    # GridSearchCV
    gs = GridSearchCV(estimator=pipes[acronym],
                      param_grid=param_grids[acronym],
                      scoring='f1_micro',
#                       n_jobs=-1,
                      cv=ps,
                      return_train_score=True)
        
    # Fit the pipeline
    gs = gs.fit(X_train_valid, y_train_valid)
    
    # Update best_score_param_estimator_gs
    best_score_param_estimator_gs.append([gs.best_score_, gs.best_params_, gs.best_estimator_])
    
    # Sort cv_results in ascending order of 'rank_test_score' and 'std_test_score'
    cv_results = pd.DataFrame.from_dict(gs.cv_results_).sort_values(by=['rank_test_score', 'std_test_score'])
    
    # Get the important columns in cv_results
    important_columns = ['rank_test_score',
                         'mean_test_score', 
                         'std_test_score', 
                         'mean_train_score', 
                         'std_train_score',
                         'mean_fit_time', 
                         'std_fit_time',                        
                         'mean_score_time', 
                         'std_score_time']
    
    # Move the important columns ahead
    cv_results = cv_results[important_columns + sorted(list(set(cv_results.columns) - set(important_columns)))]

    # Write cv_results file
    cv_results.to_csv(path_or_buf='cv_results/' + acronym + '.csv', index=False)


# ### Model Selection

# In[621]:


# Sort best_score_param_estimator_gs in descending order of the best_score_
best_score_param_estimator_gs = sorted(best_score_param_estimator_gs, key=lambda x : x[0], reverse=True)

# Print best_score_param_estimator_gs
pd.DataFrame(best_score_param_estimator_gs, columns=['best_score', 'best_param', 'best_estimator'])

print(best_score_param_estimator_gs[0])


# # Plot the Feature Importance

# The code below shows how to create the directory for the figures.

# In[602]:


import os

# Make directory
directory = os.path.dirname('./figure/')
if not os.path.exists(directory):
    os.makedirs(directory)
print(best_score_param_estimator_gs)


# The code below shows how to get the feature importance detected by random forest.

# In[603]:


# Implement me
# Get the best_score, best_param and best_estimator of random forest obtained by GridSearchCV
best_score_rfc, best_param_rfc, best_estimator_rfc = best_score_param_estimator_gs[1]

# Implement me
# Get the dataframe of feature and importance
df_fi_rfc = pd.DataFrame(np.hstack((features.reshape(-1, 1), 
                         best_estimator_rfc.steps[0][1].feature_importances_.reshape(-1, 1))),
                         columns=['Features', 'Importance'])

# Implement me
# Sort df_fi_rfc in descending order of the importance
df_fi_rfc = df_fi_rfc.sort_values(ascending=False, by='Importance').reset_index(drop=True)

# Print df_fi_rfc
print(df_fi_rfc)


# The code below shows how to create the bar plot of feature importance.

# In[604]:


# Create a figure
fig = plt.figure(figsize=(10, 5))

# The bar plot of feature importance
plt.bar(df_fi_rfc['Features'][:7], df_fi_rfc['Importance'][:7], color='green')

# Set x-axis
plt.xlabel('Features')
plt.xticks(rotation=90)

# Set y-axis
plt.ylabel('Importance')

# Save and show the figure
plt.tight_layout()
plt.savefig('./figure/feature_importance_rfc.pdf')
plt.show()


# In[605]:


# Get the best_score, best_param and best_estimator obtained by GridSearchCV
best_score_gs, best_param_gs, best_estimator_gs = best_score_param_estimator_gs[0]

# Get the prediction on the testing data using best_model
y_test_pred = best_estimator_gs.predict(X_test)

print(y_test_pred)
print(y_test)

np.count_nonzero(y_test)
y_test.shape  #249 total, 127 non zero, 122 zeros

#y_train.shape # total 796, 400 zeroes, 396 non zero


# ### Evaluating Results

# In[606]:


from sklearn import metrics
import sklearn.metrics as metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import accuracy_score
from sklearn.metrics.classification import cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

print(accuracy_score(y_test, y_test_pred)* 100)
print("ROC_AUC : ", roc_auc_score(y_test, y_test_pred) * 100)
print("Cohen Kappa: ", cohen_kappa_score(y_test, y_test_pred)* 100)

# ROC Graph
y_test_pred_score = best_estimator_gs.predict_proba(X_test)
preds = y_test_pred#y_test_pred_score[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
class_names = df[target].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

import seaborn as sns
plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)

hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.title("Confusion Matrix XGBoost Model")
# plt.tight_layout()
plt.show()

(df == 0).sum(axis=1)

df_cm


# # Generating the Submission File

# In[612]:


# Transform y_test_pred back to the original class
y_test_pred = le.inverse_transform(y_test_pred)
y_test = le.inverse_transform(y_test)

# Get the submission dataframe
df_submit = pd.DataFrame(np.hstack((y_test.reshape(-1, 1), y_test_pred.reshape(-1, 1))),
                         columns=[target, target])                                                                                      

df_submit.columns = ['treatment', 'pred_treatment']


# In[613]:


bucket_name = 'mentalhealthsurveydatabucket'
submission_file = 'result.csv'

df_submit.to_csv(submission_file)
s3 = boto3.resource('s3')
s3.meta.client.upload_file(submission_file, bucket_name, 'Data/result.csv')

os.remove(submission_file)


# In[639]:


from sklearn.externals import joblib
import tarfile
sm_boto3 = boto3.client('sagemaker')
sess = sagemaker.Session()
joblib.dump(best_estimator_gs, "best_estimator_gs.joblib")
fileadd = tarfile.open('model.tar.gz','w:gz')
fileadd.add('best_estimator_gs.joblib')
fileadd.close()
#Upload the file to S3 bucket
bucket_s3 = "mentalhealthsurveydatabucket"
sess.upload_data(path='model.tar.gz', bucket=bucket_s3,)


# In[641]:


from sagemaker import s3
down_load = s3.S3Downloader()
file = "data/model.tar.gz"
down_load.download("s3://{}/{}".format(bucket_s3,file),".")
tar = tarfile.open("model.tar.gz", "r:gz")
tar.extractall()
tar.close()
best_estimator_gs = joblib.load("best_estimator_gs.joblib")


# In[643]:


from sagemaker.amazon.amazon_estimator import get_image_uri
image_name = get_image_uri(boto3.Session().region_name, 'xgboost', '0.90-1')
role = get_execution_role()


# In[654]:


s3_model_location = 's3://mentalhealthsurveydatabucket/data/model.tar.gz'
model = sagemaker.model.Model(model_data = s3_model_location,
                              image = image_name,
                              role = role,
                              sagemaker_session = sess)


# In[659]:


predictor = gs.deploy(initial_instance_count=1, instance_type='ml.t2.medium', endpoint_name = 'xgboost-v2')


# In[647]:


y_pred1 = best_estimator_gs.predict(X_test)


# In[648]:


y_pred1


# In[657]:


y_pred2 = predictor.predict(X_test)


# In[ ]:


y_pred2

