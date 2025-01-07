from pyfaidx import Fasta
import sys
import random
import copy

#global
GENOME_FILE='GCF_003254395.2_Amel_HAv3.1_genomic.fna' #download from NCBI
#https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_003254395.2/

label_dic = {'D':'0','F':'1','N':'2','Q':'3'}


def get_window_positions(a, b, window_size, stride):
    start_points = []
    end_points = []
    for start in range(a, b - window_size + 2, stride):
        end = start + window_size
        start_points.append(start)
        end_points.append(end)
    return start_points, end_points

def get_bed_ls(line_ls,augment=True,size=100):
    global label_dic
    window_size=500
    stride=10
    bed_ls=[]
    for line in line_ls:
        m = line.strip('\n').split('\t')
        chrom = m[0].replace('-','_')
        start = int(m[1])
        end = int(m[2])
        name = "{}|{}".format(m[3],m[5])
        label = label_dic[m[4]]
        if augment:
            start_ls, end_ls = get_window_positions(start-size, end+size, window_size, stride)
            chrom_ls = [chrom] * len(start_ls)
            #Chromosome:Start-End|Gene@1_Group
            name_ls = ["{}@{}_{}".format(name,i,label) for i in range(1,len(start_ls)+1)]
            bed_ls.extend([[i,j,k,l] for i, j, k,l in zip(chrom_ls, start_ls, end_ls,name_ls)])
        else:
            name = "{}@{}_{}".format(name,0,label)
            bed_ls.append([chrom,start,end,name])
    return bed_ls

def fasta_oup(bed_ls,filename):
    global GENOME_FILE
    genome=Fasta(GENOME_FILE,key_function=lambda x: x.split(' ')[0])
    result_ls = list(map(lambda i: '>{0}\n{1}'.format(bed_ls[i][3],genome[bed_ls[i][0]][int(bed_ls[i][1]):int(bed_ls[i][2])].seq.upper()), range(len(bed_ls))))
    with open(filename, "w") as oup:
        oup.write('\n'.join(result_ls))

#INPUT
inp_filename = sys.argv[1]
data_folder = sys.argv[2]
#test
#inp_filename = 'data/test.diff.bed'


#OUTPUT
train_filename = data_folder+'/train.fa'
valid_filename = data_folder+'/validation.fa'
test_filename = data_folder+'/test.fa'


#MAIN
#shuffle
with open(inp_filename, "r") as inp:
    lines = inp.readlines()
lines = lines[1:]
random.seed(123)
random.shuffle(lines)

#size
total_lines = len(lines)
train_size = int(0.8 * total_lines)
cv_test_size = int(0.1 * total_lines)

#split dataset
#train
train_lines = lines[:train_size]
bed_ls_train = get_bed_ls(train_lines,augment=True)
random.shuffle(bed_ls_train)

#cross validation
cv_lines = lines[train_size:train_size + cv_test_size]
bed_ls_cv = get_bed_ls(cv_lines,augment=True)
random.shuffle(bed_ls_cv)

#test
test_lines = lines[train_size + cv_test_size:]
bed_ls_test = get_bed_ls(test_lines,augment=True)
random.shuffle(bed_ls_test)

#oup
fasta_oup(bed_ls_train,train_filename)
fasta_oup(bed_ls_cv,valid_filename)
fasta_oup(bed_ls_test,test_filename)







