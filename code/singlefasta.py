from Bio import SeqIO


def GetSingleFasta(overall, single):
    fasta = open(overall)
    for record in SeqIO.parse(fasta, "fasta"):  # SeqIO.parse可以逐个读取fasta格式的样本   按顺序提取每个样本的ID和seq
        name = record.name
        seq = str(record.seq)
        file_name = name.split('|')[0] + '_' + name.split('|')[1] + ".fasta"  # 按序列名字进行更改
        #file_name = name + '.fasta'
        single_fasta = open(single + '/' + file_name, 'w')
        single_fasta.write('>' + name + '\n')
        single_fasta.write(seq+'\n')


if __name__ == '__main__':
    GetSingleFasta(r'../p.fasta', r'../testp')
