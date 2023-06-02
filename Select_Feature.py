#从降维之后的训练集中提取对应的特征
import numpy as np
import pandas as pd

def GetFeatureToTest(train_file_addr, test_file_addr, filename, saveaddr):
    
    #读取特征提取后的训练集的非标签列
    train = pd.read_csv(train_file_addr)
    train_column = list(train.columns)[:-1]#[:, :-1]
    print('train:  ',train.shape)

    #print(train_column)
    test = pd.read_csv(test_file_addr)
    print('test:  ',test.shape)
    test = test.iloc[:, 0:]
    test.isna().values.any()
    test = test.dropna(axis=0) #丢弃含空值的行
    test_data = test.iloc[:, :-1]
    lable = test.iloc[:, [-1]] #标签

    output = pd.DataFrame(data=test_data, columns=train_column)
    test_new_csv = pd.concat([output, lable], axis=1)
    test_new_csv.to_csv(saveaddr + filename +'.csv', index=False)

if __name__ == '__main__':
    GetFeatureToTest(r'../data/RFECV_Train.csv', r"/data/tjzhang/luosu/LiuXuan/ACP240/csv/merge.csv.csv", 'RFECV_Test',  '/data/tjzhang/luosu/LiuXuan/ACP240/csv/')

    test = pd.read_csv(r'/data/tjzhang/luosu/LiuXuan/ACP240/feature/RFECV_Test.csv')
    print(test.shape)