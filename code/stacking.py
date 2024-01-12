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
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

import Bio

def performance(y, predicted_label, predicted_proba):
    # 分类准确率分数是指所有分类正确的百分比。metrics.accuracy_score(y_true, y_pred, *[, ...])
    accuracy = accuracy_score(y, predicted_label)  # acc
    cm = confusion_matrix(y, predicted_label)  # 混淆矩阵
    sensitivity = (cm[0][0])/(cm[0][0]+cm[1][0])
    specificity = (cm[1][1]) / (cm[1][1] + cm[0][1])
    MCC = matthews_corrcoef(y, predicted_label)  # MCC
    auc = roc_auc_score(y, predicted_proba)
    # cm = confusion_matrix(y, predicted_label)  # 混淆矩阵
    return accuracy, sensitivity, specificity, MCC, auc



def cross_validation_k(path):
    fp = open(r'../performance/train.txt', 'a')  # 用于记录训练集性能指标
   
    data_train = pd.read_csv(path)
    
    kfold = KFold(n_splits=5, shuffle=True)  # 初始化

    x_train = np.array(
        data_train.iloc[0:, :data_train.shape[1] - 1])  # 取所有行，取到倒数第二列，不要标签列

    y_train = data_train["class"]  # 训练集所有数据的标签
    # print(y_train)
    acc_sum = SN_sum = SP_sum = mcc_sum = auc_sum = 0  # 统计各种性能指标在每一折的表现之和，用于最后求平均
    tmp = 0
    i = 1

    sm = SMOTEENN()
    x_train, y_train = sm.fit_resample(x_train, y_train)

    print(x_train.shape)

    #############################################################  change modle here  ######################################################
    svm = SVC()  # soft probability=True
    rf = RandomForestClassifier()
    gdbt = GradientBoostingClassifier()
    nb = GaussianNB()
    lgbm = LGBMClassifier()
    knn = KNeighborsClassifier()
    lr = LogisticRegression()
    adaboost = AdaBoostClassifier()
    et = ExtraTreesClassifier()
    mlp = MLPClassifier()
    emodel = StackingClassifier(classifiers=[rf, lgbm, et, knn, mlp], meta_classifier= lr)
    for train_index, test_index in kfold.split(x_train):
    
        train_data, train_label = x_train[train_index], y_train[train_index]
        test_data, test_label = x_train[test_index], y_train[test_index]

        
        tmp = str(emodel).split('(')[0]
        
        emodel.fit(train_data, train_label)
        # y_predict是利用模型预测得到的class，就是对（）中的样本的预测结果
        y_predict = emodel.predict(test_data)
        y_predict_proba = emodel.predict_proba(test_data)
        y_predict_proba = pd.DataFrame((y_predict_proba[:,1].tolist()))
        acc, SN, SP, mcc, auc = performance(test_label, y_predict, y_predict_proba)
        print("第", i, "折各项评价指标为")
       
        print("SN: %.3f%%" % (SN * 100), end='   ')
        print("SP: %.3f%%" % (SP * 100), end='   ')
        print("accuracy: %.3f%%" % (acc * 100), end='   ')
        print("MCC: %.3f" % mcc, end='   ')
        print("AUC: %.3f" % auc, end='   ')
        print('\n')
        acc_sum += acc
        SN_sum += SN
        SP_sum += SP
        mcc_sum += mcc
        auc_sum += auc
        i += 1
    joblib.dump(emodel, r'../model' + '//' + tmp +
                '_et_new.pkl')  # sklearn中的joblib模块进行模型保存与加载

    print('指标各项平均：')
   
    print("SN: %.3f%%" % ((SN_sum / 10) * 100), end='   ')
    print("SP: %.3f%%" % ((SP_sum / 10) * 100), end='   ')
    print("accuracy: %.3f%%" % ((acc_sum / 10) * 100), end='   ')
    print("MCC: %.3f" % ((mcc_sum / 10)), end='   ')
    print("AUC: %.3f" % ((auc_sum / 10)), end='   ')
    print('\n')
    # 将性能指标写到文件中
    fp.write("特征名: " + str(path).split('/')[2] + ',' + " 分类器名: " + tmp + '\n')
    
    fp.write("accuracy: %.3f%%" % ((acc_sum / 10) * 100))
    fp.write("    SN: %.3f%%" % ((SN_sum / 10) * 100))
    fp.write("    SP: %.3f%%" % ((SP_sum / 10) * 100))
    fp.write("    MCC: %.3f\n" % ((mcc_sum / 10)))
    fp.write("    AUC: %.3f\n" % ((auc_sum / 10)))
    fp.write('\n')
    fp.close()
    return 0


if __name__ == '__main__':

    cross_validation_k(r'../data/RFECV_Train.csv')
