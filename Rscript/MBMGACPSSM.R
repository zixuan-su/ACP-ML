library(PSSMCOOL)
library(stringr)
filename=list.files("../testp_pssm")    #获取当前工作空间中文件的名字并形成列表
write.csv(filename,file='../csv/testp_mb.csv',row.names=FALSE)

MBMGACPSSM<- cbind(MBMGACPSSM(str_c("../testp_pssm/",filename[1])))   #按照特征提取方法去提取特征并合并（测试类代码）
print(filename[1])
for(i in 2:length(filename))
{
    MBMGACPSSM<- cbind(MBMGACPSSM,MBMGACPSSM(str_c("../testp_pssm/",filename[i])))   #循环按照特征提取方法去提取特征并合并
    print(filename[i])
}
write.csv(t(MBMGACPSSM),file='../csv/p_MBMGACPSSM.csv',row.names=FALSE)  #将其输出到一个csv文件中


filename=list.files("../testn_pssm")    #获取当前工作空间中文件的名字并形成列表
write.csv(filename,file='../csv/testn_mb.csv',row.names=FALSE)
MBMGACPSSM<- cbind(MBMGACPSSM(str_c("../testn_pssm/",filename[1])))   #按照特征提取方法去提取特征并合并（测试类代码）
for(i in 2:length(filename))
{
    MBMGACPSSM<- cbind(MBMGACPSSM,MBMGACPSSM(str_c("../testn_pssm/",filename[i])))   #循环按照特征提取方法去提取特征并合并
    print(filename[i])
}
write.csv(t(MBMGACPSSM),file='../csv/n_MBMGACPSSM.csv',row.names=FALSE)  #将其输出到一个csv文件中