from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.svm import SVC 
from sklearn.feature_selection import RFECV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data_train = pd.read_csv(r"../data/train/smoteenn_mrmd.csv")
print(data_train.shape)
# datas = np.array(df)
# datas = datas
X = np.array(
        data_train.iloc[0:, :data_train.shape[1] - 1])
y = data_train["class"]
clf_rf_4 = RandomForestClassifier() 
#clf_rf_4 = SVC()
rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5,scoring='accuracy', n_jobs= -1)   #5折交叉验证
rfecv = rfecv.fit(X, y)
print('Optimal number of features :', rfecv.n_features_) #The number of selected features with cross-validation.
X=pd.DataFrame(X)
print('Best features :', X.columns[rfecv.support_])# The mask of selected features.
lst = pd.DataFrame(list(X.columns[rfecv.support_]))
lst.to_csv("rfe.csv",index=False)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score of number of selected features")
#lst = list(rfecv.cv_results_.values())
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score']) # A dict with keys:

#plt.plot(range(1, len(rfecv.cv_results_) + 1), lst)
plt.savefig('result_plot_rf2.png', bbox_inches='tight')

