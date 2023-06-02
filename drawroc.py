import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

def Draw_ROC(file1, file2, file3, file4, file5):
    '''这里注意读取csv的编码方式，
    如果csv里有中文，在windows系统上可以直接用encoding='ANSI'，
    但是到了Mac或者Linux系统上会报错：`LookupError: unknown encoding: ansi`。
    解决方法：
    1. 可以改成encoding='gbk'；
    2. 或者把csv文件里的列名改成英文，就不用选择encoding的方式了。
    '''
    data1=pd.read_csv(file1)
    data1=pd.DataFrame(data1)
    # print(data1)

    data2=pd.read_csv(file2)
    data2=pd.DataFrame(data2)

    # data3 = pd.read_csv(file3)
    # data3 = pd.DataFrame(data3)

    data4 = pd.read_csv(file4)
    data4= pd.DataFrame(data4)

    # data5 = pd.read_csv(file5)
    # data5= pd.DataFrame(data5)

    fpr_ow,tpr_ow,thresholds=roc_curve(list(data1['class']),
                                        list(data1['proba']))
    #roc_auc_RF=auc(fpr_RF,tpr_RF)
    roc_auc_ow = roc_auc_score(list(data1['class']), list(data1['proba'])) #ourwork

    fpr_mACPpred,tpr_mACPpred,thresholds=roc_curve(list(data2['class']),
                                       list(data2['proba']))
    #roc_auc_Lgbm=auc(fpr_Lgbm,tpr_Lgbm)
    roc_auc_mACPpred = roc_auc_score(list(data2['class']), list(data2['proba']))

    # fpr_iDACP, tpr_iDACP, thresholds_DNN = roc_curve(list(data3['class']),
    #                                        list(data3['proba']))
    # # #roc_auc_Stacking = auc(fpr_Stacking, tpr_Stacking)
    # roc_auc_iDACP = roc_auc_score(list(data3['class']), list(data3['proba']))


    fpr_ACPred, tpr_ACPred, thresholds_et = roc_curve(list(data4['class']),
                                           list(data4['proba']))
    #roc_auc_Stacking = auc(fpr_Stacking, tpr_Stacking)
    roc_auc_ACPred = roc_auc_score(list(data4['class']), list(data4['proba']))

    # fpr_Voting, tpr_Voting, thresholds_DNN = roc_curve(list(data5['class']),
    #                                        list(data5['proba']))
    # roc_auc_Voting = roc_auc_score(list(data5['class']), list(data5['proba']))
    #roc_auc_Voting = auc(fpr_Voting, tpr_Voting)

    font = {'family': 'Times New Roman',
            'size': 12,
            }
    
    sns.set(font_scale=1.2)
    plt.rc('font',family='Times New Roman')

    plt.plot(fpr_ow,tpr_ow,'purple',label='OurWork_AUC = %0.3f'% roc_auc_ow)
    plt.plot(fpr_mACPpred,tpr_mACPpred,'blue',label='mACPpred_AUC = %0.3f'% roc_auc_mACPpred)
    #plt.plot(fpr_iDACP,tpr_iDACP,'red',label='iDACP_AUC = %0.3f'% roc_auc_iDACP)
    plt.plot(fpr_ACPred,tpr_ACPred,'black',label='ACPred_AUC = %0.3f'% roc_auc_ACPred)
    #plt.plot(fpr_Voting,tpr_Voting,'yellow',label='Voting(soft)_AUC = %0.3f'% roc_auc_Voting)
    
    plt.legend(loc='lower right',fontsize = 12)
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('True Positive Rate',fontsize = 14)
    plt.xlabel('Flase Positive Rate',fontsize = 14)
    plt.savefig('../roc_164.png')
    
if __name__=="__main__":
    Draw_ROC(
        r'../drawroc/acp164.csv',
        r'../drawroc/mpred.csv',
        #r'../ROCFile/iDACP_proba.csv',
        ' ',
        r'../drawroc/acpRED.csv',
        ' '
             #r'../file/soft.csv',
             )














