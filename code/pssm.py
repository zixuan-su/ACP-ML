# coding:utf-8
import os
import threading
def screen(file_path):
    num=0
    count=0
    path=file_path
    txt_list=[]
    file_list=os.listdir(path)
    for i in file_list:#检索有多少.txt文件
        file_ext=os.path.splitext(i)
        front,ext=file_ext
        if ext==".fasta":
            txt_list.append(i)
            num=num+1
    print(txt_list)
    print(num)#计算序列总数
    for i in txt_list: #批量生成PSSM矩阵
        print(i)
        print('-----------------------------------------------')
        # os.chdir(r"/home/yrjia/yrjia/Stack-Cas/fasta")
        #os.system(r"psiblast -in_msa C:/Users/Administrator/Desktop/blast/input/"+str(i)+ r"-db D:\NCBI\blast-2.3.0+\db\uniprot-all.fasta -comp_based_stats 1 -inclusion_ethresh  0.001 -num_iterations 3 -out_ascii_pssm C:\Users\Administrator\Desktop\blast\output\\"+str(i).split('.')[0]+".pssm")
        os.system('psiblast -query '+file_path+str(i)+' -db /data/liuxuan/20220315/uniref50.fasta -num_iterations 3 -out_ascii_pssm /data/liuxuan/20220315/testpssm/'+i.split('.')[0]+'.pssm')
    return 0

if __name__ =='__main__':
    screen(r"/data/liuxuan/20220315/test/")
# count_sequence_num(r"E:\blast\output")

