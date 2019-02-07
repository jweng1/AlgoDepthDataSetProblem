import pandas as pd
from sklearn.cross_validation import train_test_split
import csv
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate


mydataset = pd.read_csv('dataSet.csv')
#mFeatures and mLables
X = mydataset.iloc[:,1:15].values
y = mydataset.iloc[:,-3].values

#hot encode
labelencoder_X = LabelEncoder()
X[:,1] = labelencoder_X.fit_transform(X[:,1]) #symbol hot encode
X[:,0] = labelencoder_X.fit_transform(X[:,0]) #date hot encode
Y = pd.DataFrame(X)

#split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# classifiers = [
#     KNeighborsClassifier(),
#     SVC(),
#     DecisionTreeClassifier(),
#     RandomForestClassifier(),
#     AdaBoostClassifier(),
#     GradientBoostingClassifier(),
#     GaussianNB(),
#     LogisticRegression(),
#     LinearSVC()]
#
# for clf in classifiers:
#     name = clf.__class__.__name__
#     clf.fit(X_train, y_train)
#     prediction = clf.predict(X_test)
#     rank = pd.DataFrame(data=np.column_stack([prediction, y_test]),columns=['Predicted','Real'])
#     accurracy = np.sum(rank.Predicted.values == rank.Real.values)
#     accurracy = accurracy/len(y_test)
#     print(accurracy, name)

##result
# 0.8093548288415063 KNeighborsClassifier
# 0.5180029979503809 SVC
# 0.5903820857169079 DecisionTreeClassifier
# 0.6363914466640154 RandomForestClassifier
# 0.521704548930833 AdaBoostClassifier
# 0.5326562452201046 GradientBoostingClassifier
# 0.5042674905931659 GaussianNB
# 0.5188901465324727 LogisticRegression
# 0.5189207378628896 LinearSVC


#SVC
X, y = samples_generator.make_classification(n_informative=5, n_redundant=0, random_state=42)
anova_filter = SelectKBest(f_regression, k=5)
clf = svm.SVC(kernel='linear')
anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])
anova_svm.set_params(anova__k=10, svc__C=.01).fit(X, y)
Pipeline(memory=None,steps=[('anova', SelectKBest(...)),('svc', SVC(...))])
prediction = anova_svm.predict(X)
score = anova_svm.score(X, y)
print(score,'SVC')
print("####################")

#KNeightborsClassifer
scaler = MinMaxScaler()
knn = KNeighborsClassifier(n_neighbors=5)
scaler.fit(X_train)
X_train  = scaler.fit_transform(X_train)
knn.fit(X_train, y_train)
pipeline = Pipeline([('minmax',MinMaxScaler()),('clf',KNeighborsClassifier())])
cross_validate(pipeline,X_train,y_train,cv=5)

knn.score(X_test,y_test)
print(score, 'KNeighborsClassifier')


#------------------------------------------------------------------------------------------------------------------------------------------------
