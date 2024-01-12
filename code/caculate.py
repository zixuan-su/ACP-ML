#根据测试集在训练好的模型预测概率得到的csv文件计算各种指标

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, matthews_corrcoef
from sklearn.metrics import roc_auc_score, confusion_matrix, auc, roc_curve

def perform(path):
    data = pd.read_csv(path)
    line = line = data.shape[0]
    #data = np.array(data)
    y = data['class']
    predicted_label = data['predict']
    predicted_lable_proba = data['proba']
    
     # 分类准确率分数是指所有分类正确的百分比。metrics.accuracy_score(y_true, y_pred, *[, ...])
    accuracy = accuracy_score(y, predicted_label)  # acc
    # TP/(TP+FN)
    cm = confusion_matrix(y, predicted_label)  # 混淆矩阵
    sensitivity = (cm[0][0]) / (cm[0][0]+cm[1][0])
    specificity = (cm[1][1]) / (cm[1][1] + cm[0][1])
    MCC = matthews_corrcoef(y, predicted_label)  # MCC
    fpr,tpr,thresholds=roc_curve(y, predicted_lable_proba)
    
    auc_lable = auc(fpr, tpr)
    auc_proba = roc_auc_score(y, predicted_lable_proba)
    
   
    print("SN: %.3f%%" % (sensitivity*100), end='   ')
    print("SP: %.3f%%" % (specificity*100), end='   ')
    print("accuracy: %.3f%%" % (accuracy*100), end='   ')
    print("MCC: %.3f" % (MCC), end='   ')
    print("AUC_Lable: %.3f" % (auc_lable), end='   ')
    print("AUC_Proba: %.3f" % (auc_proba), end='   ')
    print('\n')
    #return accuracy, sensitivity, specificity, MCC, auc
    



perform(r'../drawroc/acpRED.csv')
