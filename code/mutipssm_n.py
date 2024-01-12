import os
import threading

file_list=os.listdir(r'../testn')

def pssm(path):
	
	os.system('psiblast -query '+ '/data/tjzhang/luosu/LiuXuan/ACP240/testn/'+ path +' -db /data/tjzhang/luosu/LiuXuan/ACP240/base/uniref50 -num_iterations 3 -out_ascii_pssm /data/tjzhang/luosu/LiuXuan/ACP240/testnpssm/'+path.split('.')[0]+'.pssm')


if __name__ == '__main__':
    for i in file_list:
        t1 = threading.Thread(target=pssm, args=(i,))
        t1.start()   
    