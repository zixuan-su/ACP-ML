library(PSSMCOOL)
library(stringr)

filename=list.files("../testp_pssm")    #获取当前工作空间中文件的名字并形成列表
write.csv(filename,file='../csv/testp.csv',row.names=FALSE)
CS_PSe_PSSM<- cbind(CS_PSe_PSSM(str_c("../testp_pssm/",filename[1]),'total'))   #按照特征提取方法去提取特征并合并（测试类代码）
print(filename[1])
for(i in 2:length(filename))
{
    
    print(filename[i])
    CS_PSe_PSSM<- cbind(CS_PSe_PSSM,CS_PSe_PSSM(str_c("../testp_pssm/",filename[i]),'total'))
}   #循环按照特征提取方法去提取特征并合并
write.csv(t(CS_PSe_PSSM),file='../csv/p_CS_PSe_PSSM.csv',row.names=FALSE)  #将其输出到一个csv文件中


filename=list.files("../testn_pssm")    #获取当前工作空间中文件的名字并形成列表
write.csv(filename,file='../csv/testn.csv',row.names=FALSE)
CS_PSe_PSSM<- cbind(CS_PSe_PSSM(str_c("../testn_pssm/",filename[1]),'total'))   #按照特征提取方法去提取特征并合并（测试类代码）
print(filename[1])
for(i in 2:length(filename))
{   
    print(filename[i])
    CS_PSe_PSSM<- cbind(CS_PSe_PSSM,CS_PSe_PSSM(str_c("../testn_pssm/",filename[i]),'total'))   #循环按照特征提取方法去提取特征并合并
}
write.csv(t(CS_PSe_PSSM),file='../csv/n_CS_PSe_PSSM.csv',row.names=FALSE)  #将其输出到一个csv文件中