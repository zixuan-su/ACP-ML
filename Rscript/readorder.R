#阳性阅读顺序
filename=list.files("../testp_pssm") 
write.csv(filename,file='../csv/testp.csv',row.names=FALSE)

#阴性阅读顺序
filename=list.files("../testn_pssm") 
write.csv(filename,file='../csv/testn.csv',row.names=FALSE)