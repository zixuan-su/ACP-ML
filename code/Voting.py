# SVM评估降维效果
import joblib
import numpy as np
import pandas as pd
import matplotlib as plt
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.metrics import accuracy_score, recall_score, matthews_corrcoef
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC  # SVM
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier  # GBDT 集成
from sklearn.naive_bayes import GaussianNB  # naive_bayes 使用的是GaussianNb
from sklearn.ensemble import AdaBoostClassifier  # AdaBoost  #集成
# Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier  # ET
from sklearn.ensemble import RandomForestClassifier  # RF
from sklearn.neighbors import KNeighborsClassifier  # KNN
from xgboost import XGBClassifier  # xgboost 集成
from lightgbm.sklearn import LGBMClassifier  # LightGBM
from sklearn.model_selection import GridSearchCV  # 网格搜素与交叉验证包
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn import preprocessing, metrics  # 标准化
from sklearn.decomposition import PCA  # 导入主成分分析
from imblearn.over_sampling import SMOTE, RandomOverSampler  # 过采样
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from mlxtend.classifier import StackingClassifier
#from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
def performance(y, predicted_label):
    # 分类准确率分数是指所有分类正确的百分比。metrics.accuracy_score(y_true, y_pred, *[, ...])
    accuracy = accuracy_score(y, predicted_label)  # acc
    # TP/(TP+FN)
    sensitivity = recall_score(y, predicted_label)  # SN
    # len(y)代表的是TP+TN+FN+FP,sum(y)是标签为1的个数,做差是标签为0的个数,所以总体就是TN/(TN+FP)
    specificity = (accuracy * len(y) - sensitivity *
                   sum(y)) / (len(y) - sum(y))  # SP
    MCC = matthews_corrcoef(y, predicted_label)  # MCC
    auc = roc_auc_score(y, predicted_label)
    # cm = confusion_matrix(y, predicted_label)  # 混淆矩阵
    return accuracy, sensitivity, specificity, MCC, auc

#this function sctually is a voting function  , I forgot to rename it, it's been a long time since now...... 
def Stacking (path):

    fp = open(r'../result/voting_performance.txt', 'a')  # 用于记录训练集性能指标
    #加载数据集
    data_train = pd.read_csv(path)
    x_train = np.array(
        data_train.iloc[0:, :data_train.shape[1] - 1])  
    y_train = data_train["class"]  

    #平衡数据集
    sm = SMOTEENN()
    x_train, y_train = sm.fit_resample(x_train, y_train)
    print(x_train.shape) 

    #确定一二层基分类器
    svm = SVC()  # soft probability=True
    rf = RandomForestClassifier()
    gdbt = GradientBoostingClassifier()
    nb = GaussianNB()
    lgbm = LGBMClassifier()
    knn = KNeighborsClassifier()
    lr = LogisticRegression()
    adaboost = AdaBoostClassifier()
    mlp = MLPClassifier()
    #voting = VotingClassifier(classifiers=[rf, lgbm, mlp, svm], meta_classifier= knn)
    voting = VotingClassifier(estimators=[('svc', svm), ('knn', knn), ('mlp', mlp), ('rf', rf),('gdbt', gdbt), ('lgbm', lgbm)],
                           voting='hard')
    kfold = KFold(n_splits=10, shuffle=True)  # 初始化

    acc_sum = SN_sum = SP_sum = mcc_sum = auc_sum = 0  # 统计各种性能指标在每一折的表现之和，用于最后求平均
    tmp = 0
    i = 1

    for train_index, test_index in kfold.split(x_train):
        
        train_data, train_label = x_train[train_index], y_train[train_index]
        test_data, test_label = x_train[test_index], y_train[test_index]

        voting.fit(train_data, train_label)
        
        y_predict = voting.predict(test_data)

        tmp = str(voting).split('(')[0]

        acc, SN, SP, mcc, auc = performance(test_label, y_predict)
        print("第", i, "折各项评价指标为")
        print("accuracy: %.3f%%" % (acc * 100), end='   ')
        print("SN: %.3f%%" % (SN * 100), end='   ')
        print("SP: %.3f%%" % (SP * 100), end='   ')
        print("MCC: %.3f" % mcc, end='   ')
        print("AUC: %.3f" % auc, end='   ')
        print('\n')
        acc_sum += acc
        SN_sum += SN
        SP_sum += SP
        mcc_sum += mcc
        auc_sum += auc
        i += 1
    # joblib.dump(emodel, r'../model' + '//' + tmp +
    #             '_MRMD.pkl')  # sklearn中的joblib模块进行模型保存与加载
    joblib.dump(voting, r'../finalmodel' + '//' + tmp +
                'hard_MRMD.pkl')
    print('指标各项平均：')
    print("accuracy: %.3f%%" % ((acc_sum / 10) * 100), end='   ')
    print("SN: %.3f%%" % ((SN_sum / 10) * 100), end='   ')
    print("SP: %.3f%%" % ((SP_sum / 10) * 100), end='   ')
    print("MCC: %.3f" % ((mcc_sum / 10)), end='   ')
    print("AUC: %.3f" % ((auc_sum / 10)), end='   ')
    print('\n')

    fp.write("特征名: " + str(path).split('/')[2] + ',' + " 分类器名: " + tmp + '\n')
    # fp.write("特征名: " + 'ap' + ',' + " 分类器名: " + tmp + '\n')
    fp.write("accuracy: %.3f%%" % ((acc_sum / 10) * 100))
    fp.write("    SN: %.3f%%" % ((SN_sum / 10) * 100))
    fp.write("    SP: %.3f%%" % ((SP_sum / 10) * 100))
    fp.write("    MCC: %.3f\n" % ((mcc_sum / 10)))
    fp.write("    AUC: %.3f\n" % ((auc_sum / 10)))
    fp.write('\n')
    fp.close()
    return 0
if __name__ == '__main__':

    Stacking(r'../data/MRMD_Select.csv')
