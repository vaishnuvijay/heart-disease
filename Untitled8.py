#!/usr/bin/env python
# coding: utf-8

# # Heart disease

# ![](https://www.labiotech.eu/wp-content/uploads/2023/05/Cure-for-cardiovascular-diseases.jpg)

# In[1]:


#importing dataset
import pandas as pd
a=pd.read_csv("C:/Users/vaish/OneDrive/Desktop/hh.csv")
a


# In[2]:


a.describe()


# In[3]:


a.isnull().sum()


# In[4]:


a.info()


# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt
z=a.corr()
plt.figure(figsize=(10,10))
sns.heatmap(z,annot=True)


# In[6]:


plt.figure(figsize=(10,25))
sns.displot(data=a,x="age",kde=True,color="red",facecolor="cyan")
plt.title("age",fontsize=20,color="violet")


# In[7]:


plt.title("Box Plot of target",fontsize=20)
sns.boxplot(a["target"])


# In[8]:


sns.displot(data=a,x="age",col="target",kind="kde")


# In[9]:


sns.scatterplot(data=a,x="age",y="chol",hue="target")
plt.title("scatter diagram")


# In[10]:


sns.scatterplot(data=a,x="age",y="thalach",hue="target")
plt.title("scatter diagram")


# In[11]:


la={1:"postive",0:"negative"}
a["target"]=a["target"].map(la)
la1=a.target.unique()
la1


# In[12]:


plt.figure(figsize=(18,6),facecolor="pink")
plt.subplot(1,2,1)
sns.countplot(x=a["target"],color="green")
plt.title("countplot",fontsize=20)
plt.subplot(1,2,2)
plt.pie(x=a["target"].value_counts(),labels=la1,autopct="%1.2f%%",colors=["red","yellow","green"])
plt.title("pie diagram",fontsize=20)


# In[13]:


a


# In[14]:


from sklearn import preprocessing
label=preprocessing.LabelEncoder()
h=label.fit_transform(a["target"])


# In[15]:


for i,j in enumerate(label.classes_):
    print(i,j)


# In[16]:


a["target"]=h
a


# In[17]:


y=a.iloc[:,-1]
x=a.iloc[:,:-1]


# # Logistic Regression

# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
x_train,x_test,y_train,y_test=train_test_split(x,y)
reg=LogisticRegression()
reg.fit(x_train,y_train)
rs=reg.predict(x_test)
rs


# In[19]:


from collections import Counter
Counter(rs)


# In[20]:


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,ConfusionMatrixDisplay
#training score
y1=reg.score(x_train,y_train)
print("training accuracy of logistic regression is ",y1)


# In[21]:


#model accuracy
x=accuracy_score(rs,y_test)
print("model accuracy of logistic regression is ",x)


# In[22]:


confusion_matrix(y_test,rs)


# In[23]:


T=ConfusionMatrixDisplay(confusion_matrix(y_test,rs))
T.plot()


# # Random Forest Classifier

# In[24]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
pre=rf.predict(x_test)
pre


# In[25]:


y2=rf.score(x_train,y_train)
print("training accuracy of rfc is ",y2)


# In[26]:


x3=accuracy_score(pre,y_test)
print("model accuracy of rfc is ",x3)


# In[27]:


confusion_matrix(y_test,pre)
t=ConfusionMatrixDisplay(confusion_matrix(y_test,pre))
t.plot()


# In[28]:


from sklearn.svm import SVC
svm=SVC()
svm.fit(x_train,y_train)
pr1=svm.predict(x_test)
pr1


# In[29]:


y3=svm.score(x_train,y_train)
print("training accuracy of svm is ",y3)


# In[30]:


x1=accuracy_score(pr1,y_test)
print("model accuracy of svm is ",x1)


# In[31]:


print(classification_report(y_test,pr1))


# In[32]:


confusion_matrix(y_test,pr1)


# In[33]:


u=ConfusionMatrixDisplay(confusion_matrix(y_test,pr1))
u.plot()


# # Naive bayes

# In[34]:


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(x_train,y_train)
predict2=gnb.predict(x_test)
predict2


# In[35]:


y4=gnb.score(x_train,y_train)
print("training accuracy of gaussaian NB is ",y4)


# In[36]:


x4=accuracy_score(predict2,y_test)
print("model accuracy of gnb is ",x4)


# In[37]:


print(classification_report(y_test,predict2))


# In[38]:


confusion_matrix(y_test,predict2)


# In[40]:


v=ConfusionMatrixDisplay(confusion_matrix(y_test,predict2))
v.plot()


# # KNN

# In[41]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train,y_train)
predict3=knn.predict(x_test)
predict3


# In[42]:


y5=knn.score(x_train,y_train)
print("training accuracy of KNN is ",y5)


# In[43]:


x5=accuracy_score(predict3,y_test)
print("model accuracy of knn is ",x5)


# In[44]:


print(classification_report(y_test,pr1))


# In[45]:


confusion_matrix(y_test,pr1)


# In[46]:


w=ConfusionMatrixDisplay(confusion_matrix(y_test,predict3))
w.plot()


# In[47]:


import pandas as pd
o={"model":["logistic regrssion","random forest","svm","naive bayes","knn"],"accuracy score":[y1,y2,y3,y4,y5],"model accuracy":[x,x3,x1,x4,x5]}
v=pd.DataFrame(o)


# In[48]:


v.sort_values(by="accuracy score",ascending =False).style.background_gradient(cmap="cubehelix")


# In[49]:


#Random forest is more accurate 


# In[ ]:




