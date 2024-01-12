#根据训练模型验证在测试集上的效果，并将概率保存为CSV文件

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, matthews_corrcoef, confusion_matrix, roc_auc_score


def performance(y, predicted_label, predicted_prob):
    # 分类准确率分数是指所有分类正确的百分比。metrics.accuracy_score(y_true, y_pred, *[, ...])
    accuracy = accuracy_score(y, predicted_label)  # acc
    cm = confusion_matrix(y, predicted_label)  # 混淆矩阵
    sensitivity = (cm[0][0])/(cm[0][0]+cm[1][0])
    specificity = (cm[1][1]) / (cm[1][1] + cm[0][1])
    MCC = matthews_corrcoef(y, predicted_label)  # MCC
    
    #roc_auc_score()第二个传的的预测概率
    auc = roc_auc_score(y, predicted_prob)
    return accuracy, sensitivity, specificity, MCC , auc

    #return accuracy, sensitivity, specificity, MCC #hard

def test_dataset(path):
    fp = open(r'test.txt', 'a')  # 用于记录训练集性能指标
    feature_test = pd.read_csv(path)  # 固定特征文件
    
   
    y_data = np.array(feature_test.iloc[0:, :feature_test.shape[1] - 1])  # 数据
   
    y_lable = feature_test['class']
   
    model = joblib.load(r'VotingClassifiersoft.pkl')
   
    y_predict_class = model.predict(y_data)
    y_predict_probo = model.predict_proba(y_data)
    
    lst_1 = y_predict_class.tolist() #预测标签
    lst_2 = y_predict_probo[:,1].tolist() #预测概率

    lst_1 = pd.DataFrame(lst_1)
    lst_2 = pd.DataFrame(lst_2)

    #tmp 里面第一列为预测标签 第二列为预测概率 第三列为真实标签
    tmp = [lst_1, lst_2, y_lable]
    #tmp = [lst_1, y_lable]#hard
    final = pd.concat(tmp, axis=1)
    
    
    final.to_csv('acp240.csv', header=None, index=None)

    acc, SN, SP, mcc, auc_score = performance(y_lable, y_predict_class, y_predict_probo)
    #acc, SN, SP, mcc = performance(y_lable, y_predict_class, '')#hard
    print('-----------------在测试集上的性能为-----------------------------')
    print("accuracy: %.3f%%" % (acc * 100), end='   ')
    print("SN: %.3f%%" % (SN * 100), end='   ')
    print("SP: %.3f%%" % (SP * 100), end='   ')
    print("MCC: %.3f" % mcc, end='   ')
    print("AUC: %.3f" % auc_score, end='   ')
    print('\n')

    fp.write("特征名: " + str(path).split('/')[2] + ',' + " 分类器名_soft: " + str(model).split('(')[0] + '\n')
   
    fp.write("    SN: %.3f%%" % (SN * 100))
    fp.write("    SP: %.3f%%" % (SP * 100))
    fp.write("    accuracy: %.3f%%" % (acc * 100))
    fp.write("    MCC: %.3f\n" % (mcc))
    #fp.write("    AUC: %.3f\n" % (auc_score))
    fp.write('\n')
    fp.close()
    return 0
if __name__ == '__main__':
 
    test_dataset(r'RFECV_Test.csv')
  