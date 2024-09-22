#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# **데이터 구조 파악

# In[2]:


df = pd.read_csv('online_gaming_behavior.csv')
df.shape


# In[3]:


df.info()


# In[4]:


df.describe()


# **데이터 전처리

# In[5]:


sns.histplot(df['Age'], bins = 4, kde = True)


# In[6]:


sns.histplot(df['PlayTimeHours'], kde = True)


# In[7]:


sns.histplot(df['AvgSessionDurationMinutes'], kde = True)


# In[8]:


sns.histplot(df['AchievementsUnlocked'], kde = True)


# In[9]:


sns.histplot(df['PlayerLevel'], kde = True)


# In[10]:


plt.figure(figsize = (20,10))

for col in [['Age', 'PlayTimeHours', 'SessionsPerWeek', 'AvgSessionDurationMinutes', 'AchievementsUnlocked']]:
    sns.boxplot(df[col])
    plt.xlabel('features')
    plt.title(f'Box Plot of {col}')
    plt.show()


# In[11]:


for col in [['Age', 'PlayTimeHours', 'SessionsPerWeek', 'AvgSessionDurationMinutes']]:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - 1.5*iqr
    upper_limit = q3 + 1.5*iqr
    condition1 = df[col] < lower_limit
    condition2 = df[col] > upper_limit


# In[12]:


df['Age_Group'] = pd.cut(df['Age'], bins = [10, 20, 30, 40, 50], right = False, 
                         labels = ['10s', '20s', '30s', '40s'])

df['Age_Group'] = df['Age_Group'].astype('str')


# In[13]:


def Engagement_Rank (EngagementLevel):
    if EngagementLevel == 'High':
        return 3
    elif EngagementLevel == 'Medium':
        return 2
    else:
        return 1
    
df['Engagement_Rank'] = df['EngagementLevel'].apply(Engagement_Rank)


# In[14]:


def Genre_Category (Genre):
    if Genre == 'Action':
        return 1
    elif Genre == 'RPG':
        return 2
    elif Genre == 'Simulation':
        return 3
    elif Genre == 'Sports':
        return 4
    elif Genre == 'Strategy':
        return 5
    
df['Genre_Categry'] = df['GameGenre'].apply(Genre_Category)


# **변수 상관관계 분석

# In[15]:


df_for_corr = df.drop(columns = ['PlayerID', 'Gender', 'Location', 'GameGenre', 'GameDifficulty', 'EngagementLevel', 'Age_Group'])
df_for_corr.head()


# In[17]:


Corr_Matrix = df_for_corr.corr()


# In[18]:


plt.figure(figsize = (10, 6))
sns.heatmap(Corr_Matrix, vmax = 1, vmin = -1, center = 0, fmt = '0.3f', annot = True, cmap = 'coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[19]:


df_corr = Corr_Matrix['InGamePurchases'].drop('InGamePurchases').sort_values(ascending = False)
df_corr


# In[20]:


plt.figure(figsize = (10,6))
sns.barplot(data = df_corr.reset_index(), x = 'index', y = 'InGamePurchases', hue = 'index', palette = 'coolwarm', legend = False)
plt.xlabel('Feature')
plt.ylabel('InGamePurchases')
plt.title('Correlation Efficient of InGamePurchases')
plt.xticks(rotation = 45)
plt.show()


# **카테고리 별 데이터 분석 (Age)

# In[21]:


df.head()


# In[22]:


age_group_purchase_df = df.groupby('Age_Group')['InGamePurchases'].mean().reset_index()
age_group_purchase_df


# **카테고리 별 데이터 분석 (Genre)

# In[23]:


genre_group_purchase_df = df.groupby('GameGenre')['InGamePurchases'].mean().reset_index()
genre_group_purchase_df


# **카테고리 별 데이터 분석 (Age, Genre)

# In[24]:


age_genre_group_ptime_df = df.groupby(['Age_Group','GameGenre'])['InGamePurchases'].mean().reset_index()
age_genre_group_ptime_df


# In[25]:


plt.figure(figsize = (10, 6))
palette3 = sns.color_palette('coolwarm', 10)
sns.barplot(data = age_genre_group_ptime_df, x = 'Age_Group', y = 'InGamePurchases', hue = 'GameGenre', palette= palette3)
sns.set(style="whitegrid", color_codes=True)
plt.legend(loc='lower right')


# **카테고리 별 데이터 분석 (Location, Genre)

# In[26]:


loc_genre_group_igp_df = df.groupby(['Location','GameGenre'])['InGamePurchases'].mean().reset_index()
loc_genre_group_igp_df 


# In[27]:


plt.figure(figsize = (10, 6))
sns.barplot(data = loc_genre_group_igp_df, x = 'Location', y = 'InGamePurchases', hue = 'GameGenre', palette= palette3)
sns.set(style="whitegrid", color_codes=True)
plt.legend(loc='lower right')


# **랜덤포레스트 모델 (인게임 구매 예측)

# In[28]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[29]:


X = df.drop(columns = ['PlayerID', 'Location', 'GameGenre', 'Gender', 'InGamePurchases', 'GameDifficulty', 'EngagementLevel', 'Age_Group'])
y = df['InGamePurchases']


# In[30]:


X


# In[31]:


y.value_counts()


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)


# **데이터 불균형 해결: 가중치 부가

# In[33]:


model = RandomForestClassifier(n_estimators = 100, max_depth = 4, class_weight = 'balanced')


# In[34]:


model.fit(X_train, y_train)


# In[35]:


y_predict = model.predict(X_test)
y_predict


# In[36]:


from sklearn.metrics import classification_report 

print(classification_report(y_test, y_predict))


# In[37]:


importance = model.feature_importances_
importance


# In[38]:


data_sorted = np.argsort(importance) #np.argsort는 데이터의 순서를 잡아줌 

plt.figure(figsize = (10, 6))
plt.title('feature Importance')
plt.bar(range(len(importance)), importance[data_sorted]) #np.argsort를 원래 데이터에 적용하면 크기 순서로 반영
plt.xticks(range(len(importance)), X.columns[data_sorted], rotation = 90) 


# **데이터 불균형 해결: SMOTE

# In[39]:


from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state = 0)


# In[40]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)
X_train_over, y_train_over = smote.fit_resample(X_train, y_train) 


# In[41]:


print('SMOTE 적용전', X_train.shape, y_train.shape)
print('SMOTE 적용후', X_train_over.shape, y_train_over.shape)


# In[42]:


model = RandomForestClassifier(n_estimators = 100, max_depth = 4)
model.fit(X_train_over, y_train_over)


# In[43]:


y_predict = model.predict(X_test)
y_predict


# In[44]:


from sklearn.metrics import classification_report 

print(classification_report(y_test, y_predict))


# In[45]:


importance = model.feature_importances_
importance


# In[46]:


data_sorted = np.argsort(importance) #np.argsort는 데이터의 순서를 잡아줌 

plt.figure(figsize = (10, 6))
plt.title('feature Importance')
plt.bar(range(len(importance)), importance[data_sorted]) #np.argsort를 원래 데이터에 적용하면 크기 순서로 반영
plt.xticks(range(len(importance)), X.columns[data_sorted], rotation = 90) 


# **하이퍼파라미터

# In[47]:


from sklearn.model_selection import GridSearchCV


# In[48]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)


# In[49]:


# 하이퍼 파라미터 초기 설정
param_grid = {
    'n_estimators': [50, 100, 200], #랜덤포레스트의 갯수
    'max_depth': [5, 10, 15], #깊이
}


# In[50]:


model = RandomForestClassifier(random_state = 42, class_weight = 'balanced')


# In[51]:


grid_search = GridSearchCV(model, param_grid, cv = 5)


# In[52]:


grid_search.fit(X_train, y_train)


# In[54]:


best_model = grid_search.best_estimator_


# In[55]:


y_predict = best_model.predict(X_test)
y_predict


# In[56]:


from sklearn.metrics import classification_report 

print(classification_report(y_test, y_predict))


# In[57]:


importance = best_model.feature_importances_
importance


# In[58]:


data_sorted = np.argsort(importance) #np.argsort는 데이터의 순서를 잡아줌 

plt.figure(figsize = (10, 6))
plt.title('feature Importance')
plt.bar(range(len(importance)), importance[data_sorted]) #np.argsort를 원래 데이터에 적용하면 크기 순서로 반영
plt.xticks(range(len(importance)), X.columns[data_sorted], rotation = 90) 


# **로지스틱 회귀 (분류)

# In[59]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state = 0)


# In[60]:


X = df.drop(columns = ['PlayerID', 'Location', 'GameGenre', 'Gender', 'InGamePurchases', 'GameDifficulty', 'EngagementLevel', 'Age_Group'])
y = df['InGamePurchases']


# In[61]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)
X_train_over, y_train_over = smote.fit_resample(X_train, y_train) 


# In[62]:


X_train_over.shape


# In[63]:


y_train_over.value_counts()


# In[64]:


model = LogisticRegression()


# In[65]:


model.fit(X_train_over, y_train_over)


# In[66]:


y_pred = model.predict(X_test)
y_pred


# In[67]:


from sklearn.metrics import classification_report 

print(classification_report(y_test, y_pred))


# In[68]:


params = {'penalty':['l2', 'l1'], 'C': [0.01, 0.1, 1, 5, 10]}


# In[69]:


from sklearn.model_selection import GridSearchCV

grid_clf = GridSearchCV(model, param_grid = params, cv = 3) 


# In[70]:


grid_clf.fit(X_train_over, y_train_over)


# In[71]:


best_model = grid_clf.best_estimator_


# In[72]:


y_pred = best_model.predict(X_test)
y_pred


# In[73]:


from sklearn.metrics import classification_report 

print(classification_report(y_test, y_pred))

