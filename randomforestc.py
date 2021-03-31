#Import ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV,train_test_split,cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import plotly.express as px
from sklearn import metrics
import time
import warnings
from sklearn.metrics import classification_report,confusion_matrix
-------------------------------------------------------------------
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
--------------------------------------------------------		
import glob ##data read
path =r'/kaggle/input/sabrina/Data_from_Abujar_Sir' 
allFiles = glob.glob(path + "/*.csv")
frame = pd.DataFrame() 
list_ = [] 
for file_ in allFiles: 
    df = pd.read_csv(file_,index_col=None, header=0) 
    list_.append(df) 
    frame = pd.concat(list_)
	-------------------------------------------------
	from sklearn import preprocessing
le = preprocessing.LabelEncoder()
# columns to select for encoding
selected_col = ['Time']
le.fit(df[selected_col].values.flatten())
df[selected_col] = df[selected_col].apply(le.fit_transform)
------------------------------------------------------
Visualization:

corr = df.corr()          ##correlation
fig, ax = plt.subplots(figsize=(10, 8)) #figuresize
#Generate Heat Map
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f")
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.show()
-------------------------------------------------
plt.figure(figsize=(20,14))
sns.heatmap(df.corr(),annot=True,linecolor='black',linewidths=3,cmap = 'plasma')
------------------------------------------------------------------------
#Plotting All the 15 features distribution
i=1
plt.figure(figsize=(25,20))
for c in df.describe().columns[:]:
    plt.subplot(5,3,i)
    plt.title(f"Histogram of {c}",fontsize=10)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    plt.hist(df[c],bins=20,color='pink',edgecolor='k')
    i+=1
plt.show()
-------------------------------------------------------------
fig = plt.figure(figsize=(12,4))

ax=fig.add_subplot(121)
sns.distplot(df['Pos'],bins=50,color='r',ax=ax)
ax.set_title('Distribution of data pos')

ax=fig.add_subplot(122)
sns.distplot(np.log10(df['Pos']),bins=40,color='b',ax=ax)
ax.set_title('Distribution of data Poses in $log$ sacle')
ax.set_xscale('log')
----------------------------------------------------------------
plt.figure(figsize=(12,6))
sns.set_style('darkgrid')
sns.boxplot(x='ID', y='Pos', data = df, palette='OrRd', hue='ID')
sns.despine(left=True)
--------------------------------------------------------------
df=df.sample(frac=1).reset_index(drop=True)
----------------------------------------------------------
RANDOM FOREST CLASSIFIER :

y= a['Pos']
X = a.drop(['Pos'],axis = 1)
-------------------------------------------------
rfc =RandomForestClassifier(n_estimators=100)
rfc.fit(X,y)
-------------------------------------------------
imp_features = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis = 0)
 
plt.figure(figsize = (15,8))
plt.bar(X.columns, std, color = 'blue') 
plt.xlabel('Feature Labels') 
plt.ylabel('Feature Importances') 
plt.title('Comparison of different Feature Importances') 
plt.show()
-------------------------------------------------------
#Data Cleaning
drop_cols = ['Time','UNIX_T','Lux','Temp','Si','Co','Ro']
a = a.drop(drop_cols, axis=1)
----------------------------------------------------
y= a['Pos']

X = a.drop(['Pos'],axis = 1)
-------------------------------------------------
# splitting the dataset into train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2021)
-----------------------------------------------------------------
print("Shape of X_train: ",X_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of X_train: ",X_train.shape)
print("Shape of X_test: ", X_test.shape)
------------------------------------------------------------
from sklearn.preprocessing import RobustScaler, StandardScaler
# Feature Scaling
sc =StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
-------------------------------------------------------------
rfc = RandomForestClassifier(n_estimators=10)
training_start = time.perf_counter()
rfc.fit(X_train, y_train)
training_end = time.perf_counter()
prediction_start = time.perf_counter()
preds = rfc.predict(X_test)
prediction_end = time.perf_counter()
acc_rfc = (preds == y_test).sum().astype(float) / len(preds)*100
rfc_train_time = training_end-training_start
rfc_prediction_time = prediction_end-prediction_start
print("Scikit-Learn's Random Forest Classifier's prediction accuracy is: %3.2f" % (acc_rfc))
print("Time consumed for training: %4.3f seconds" % (rfc_train_time))
print("Time consumed for prediction: %6.5f seconds" % (rfc_prediction_time))
---------------------------------------------------------------------------------------
print("Scikit-Learn's Random Forest Classifier's prediction accuracy is: %3.2f" % (acc_rfc))
---------------------------------------------------------------------------------------------
Cross validation :

rfc_cv = RandomForestClassifier(max_depth=10, n_estimators=100)

scores = cross_val_score(rfc_cv, X_train, y_train, cv=5, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from sklearn.model_selection import train_test_split, cross_val_predict
predictions = cross_val_predict(rfc_cv, df.drop('Pos', axis=1), df['Pos'], cv=5)
confusion_matrix(df['Pos'], predictions)

print("Precision:", precision_score(df['Pos'], predictions, average='micro'))

print("Recall:",recall_score(df['Pos'], predictions, average='micro'))

print("F1-Score:", f1_score(df['Pos'], predictions, average='micro'))