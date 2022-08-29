# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 16:09:59 2022

@author: avidvans3


"""

"""
chocolate: Does it contain chocolate?
fruity: Is it fruit flavored?
caramel: Is there caramel in the candy?
peanutalmondy: Does it contain peanuts, peanut butter or almonds?
nougat: Does it contain nougat?
crispedricewafer: Does it contain crisped rice, wafers, or a cookie component?
hard: Is it a hard candy?
bar: Is it a candy bar?
pluribus: Is it one of many candies in a bag or box?
sugarpercent: The percentile of sugar it falls under within the data set.
pricepercent: The unit price percentile compared to the rest of the set.
winpercent: The overall win percentage according to 269,000 matchups.
"""
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

data=pd.read_csv('candydata.csv')

# data=data.drop('Index',axis=1)

# data=data[data['Avg. Session Length'].str.contains("nan")==False]

data=pd.DataFrame(data=data)

df = data.apply (pd.to_numeric, errors='coerce')

df.dropna(how='all')

# sns.countplot(x='chocolate', data=df)

"Is likeability relatd to chocolate presence ?"

col1=df['chocolate'];
col2=df['winpercent']
df2=pd.DataFrame()
df2['chocolate']=df['chocolate']
df2['winpercent']=df['winpercent']
print(df2.corr())

"Can we predict likeabikity with chocolate presence and other varibles ?"

from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import train_test_split
from statsmodels.graphics.gofplots import ProbPlot

X=pd.DataFrame()
X=df.drop('competitorname',axis=1)
X=df.drop('winpercent',axis=1)
X=X.drop('competitorname',axis=1)

Y=df['winpercent']
print(X.describe())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


X_train_const = sm.add_constant(X_train)
est = sm.OLS(Y_train, X_train_const)
est2 = est.fit()
print(est2.summary())
sns.histplot(est2.resid);
plt.xlabel('Residuals');plt.ylabel('Count')




model_fitted_y = est2.fittedvalues
# model residuals
model_residuals = est2.resid
# normalized residuals
model_norm_residuals = est2.get_influence().resid_studentized_internal
# absolute squared normalized residuals
model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
# absolute residuals
model_abs_resid = np.abs(model_residuals)
# leverage, from statsmodels internals
model_leverage = est2.get_influence().hat_matrix_diag
# cook's distance, from statsmodels internals
model_cooks = est2.get_influence().cooks_distance[0]


plt.figure()
plt.scatter(est2.fittedvalues,est2.resid)

plt.figure()
plt.scatter(est2.fittedvalues,model_norm_residuals)
plt.xlabel('Fitted values');plt.ylabel('Normalized residual')

X_test_const = sm.add_constant(X_test)

y_pred = est2.predict(X_test_const)
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))

"Does adding chocolate increase price ?"

col1=df['chocolate'];
col2=df['pricepercent']
df3=pd.DataFrame()
df3['chocolate']=df['chocolate']
df3['winpercent']=df['pricepercent']
print(df3.corr())

"Does adding chocolate increase price"

X_const_choc = sm.add_constant(X['chocolate'])
est = sm.OLS(df['pricepercent'], X_const_choc)
est3 = est.fit()
print(est3.summary())

"Can we predict chocolate presence using Logistic regression ?"

X_log=pd.DataFrame()
X_log=df.drop('competitorname',axis=1)
X_log=df.drop('chocolate',axis=1)
X_log=X_log.drop('competitorname',axis=1)

Y_log=df['chocolate']

from sklearn.linear_model import LogisticRegression
X_log_train, X_log_test, Y_log_train, Y_log_test = train_test_split(X_log, Y_log, test_size=0.33, random_state=42)


clf = LogisticRegression(random_state=0).fit(X_log_train, Y_log_train)

Y_log_pred=clf.predict(X_log_test)
# print(clf.score(X_log_train,Y_log_train))
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm=confusion_matrix(Y_log_test, Y_log_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show() 

"Lets do the same but this time decision trees"
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_log_train,Y_log_train)

Y_dec_pred=clf.predict(X_log_test)

cm=confusion_matrix(Y_log_test, Y_dec_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show() 

"Decision tree regressors vs Linear regressors ?"

from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X_train,Y_train)
Y_pred_dec=regr.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred_dec))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred_dec))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred_dec)))



