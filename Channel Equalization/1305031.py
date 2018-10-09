import math
import numpy as np
import collections

class ReadFiles:
    def __init__(self,config_f, train_f,test_f):
        self.config_f = config_f
        self.train_f = train_f
        self.test_f = test_f
        self.bits_under_window = 0
        self.clusters = 0
        self.weights = []
        self.variance = 0
        self.variance_sqrt = 0
        self.train_str = ""
        self.test_str = ""


    def read_(self,f_name):
        with open(f_name) as f:
            content = f.readlines()
        lines = [x.strip() for x in content]
        return lines[0]
    def train_test_read(self):

        self.train_str = self.read_(self.train_f)

        self.test_str = self.read_(self.test_f)
        return  self.train_str,self.test_str

    def configuration_input(self):

        with open(self.config_f) as f:
            content = f.readlines()
        lines = [x.strip() for x in content]


        firstLine = lines[0]
        firstLine = firstLine.split(" ")
        self.bits_under_window = int(firstLine[0])
        print("n: ",self.bits_under_window)

        weight_line = lines[1]
        weight_line = weight_line.split(" ")
        self.weights = [float(w) for w in weight_line]
        print("weights: ",self.weights)

        self.variance = float(lines[2])
        print("variance of noise ",self.variance)
        self.variance_sqrt  = math.sqrt(self.variance)
        self.clusters =  pow(2,self.bits_under_window)
        print("clusters :",self.clusters)


        return  self.bits_under_window,self.weights, self.clusters,self.variance,self.variance_sqrt



class Prior_determination:
    def __init__(self,bit_per_window,clusters,train_db):
        self.bit_per_window = bit_per_window
        self.clusters = clusters
        self.train_db = train_db
        self.dict_index = {}

    def init_dict(self):
        self.dict_index = {'000': 0,'001': 1,'010': 2,'011': 3,'100':4,'101': 5,'110': 6,'111': 7}

    def print_prior(self,prior_list2, limit):
        print()
        print("PRIOR PROB: ")
        for i in range(limit):
            print('cluster prior %d : %f '%((i+1),prior_list2[i]), end='', flush=True)
        print()
        print()

    def prior_prob2(self):
        prior_list = [0 for i in range(self.clusters)]
        limit = len(self.train_db)-self.bit_per_window+1

        str_list = [self.train_db[i:i+self.bit_per_window] for i in range(limit)]

        c = collections.Counter(str_list)
        str_list = list(set(str_list))

        limit = len(str_list)

        self.init_dict()

        for i in range(limit):
            cluster_no = self.dict_index[str_list[i]]
            prior_list[cluster_no] =c[str_list[i]]

        sum_ = sum(prior_list)
        prior_list2 = [prior_list[i]/sum_ for i in range(self.clusters)]

        self.print_prior(prior_list2,limit)

        return prior_list2


class Transition_states:
    def __init__(self,train_db, bits_per_n, cluster_no):
        self.train_db =train_db
        self.bits_per_n =bits_per_n
        self.cluster_no=cluster_no
        self.dict_index = {}

    def init_dict(self):
        self.dict_index = {'000': 0,'001': 1,'010': 2,'011': 3,'100':4,'101': 5,'110': 6,'111': 7}

    def print_trainsition_table(self,table_transition):
        print()
        print("Transition Table: ")
        [print(table_transition[i]) for i in range(self.cluster_no)]
        print()
        print()

    def transition_table2(self):
        table_transition =  [[0 for i in range(self.cluster_no)] for j in range(self.cluster_no)]
        self.init_dict()
        limit = len(self.train_db)-self.bits_per_n+1
        str_list = [self.train_db[i:i+self.bits_per_n] for i in range(limit)]
        limit -=1

        index_list = []

        for i in range(limit):
            present_str = str_list[i]
            present_index =  self.dict_index[present_str]
            next_str = str_list[i+1]
            next_index = self.dict_index[next_str]
            temp_list = []
            temp_list.append(present_index)
            temp_list.append(next_index)
            index_list.append(temp_list)

        limit = len(index_list)
        for i in range(limit):
            table_transition[index_list[i][0]][index_list[i][1]] +=1

        sumPerRow = [sum(table_transition[i]) for i in range(self.cluster_no)]
        #row onujayi gun keno kori?

        table_transition = [[table_transition[i][j]/sumPerRow[i] for j in range(self.cluster_no)] for i in range(self.cluster_no)]


        self.print_trainsition_table(table_transition)
        return table_transition

class Clustering:
    def __init__(self,cluster_no, train_db, bit_n,variance_sqrt,weights_):
        self.cluster_no=cluster_no
        self.train_db=train_db
        self.bit_n=bit_n
        self.variance_sqrt=variance_sqrt
        self.weights_ = weights_
        self.dict_index = {}

    def init_dict(self):
        self.dict_index = {'000': 0,'001': 1,'010': 2,'011': 3,'100':4,'101': 5,'110': 6,'111': 7}

    def determine_x(self,string_):
        x_terms = [int(string_[j])*self.weights_[j] for j in range(self.bit_n)]
        x_terms.append(np.random.normal(loc=0,scale=self.variance_sqrt))
        return sum(x_terms)

    def print_std_mean(self,std_mean_array):
        print()
        print("MEAN and STD for each cluster: ")
        [print(std_mean_array[i]) for i in range(self.cluster_no)]
        print()
        print()
    def std_mean_cluster(self):
        np.random.seed(10000)
        x_per_cluster = [[] for i in range(self.cluster_no)]

        self.init_dict()
        limit = len(self.train_db)-self.bit_n+1
        str_list = [self.train_db[i:i+self.bit_n] for i in range(limit)]

        for i in range(limit):
            x_per_cluster[self.dict_index[str_list[i]]].append(self.determine_x(str_list[i]))

        std_mean_array = [[np.mean(x_per_cluster[i]),np.std(x_per_cluster[i])] for i in range(self.cluster_no)]

        self.print_std_mean(std_mean_array)

        return std_mean_array

class Equalizer_Input:
    def __init__(self,test_db, bit_n, weights_,variance_):
        self.test_db = test_db
        self.bit_n = bit_n
        self.weights_ = weights_
        self.variance_ = variance_
        self.variance_sqrt = math.sqrt(self.variance_)

    def determine_x(self,string_):
        x_terms = [int(string_[j])*self.weights_[j] for j in range(self.bit_n)]
        x_terms.append(np.random.normal(loc=0,scale=self.variance_sqrt))
        return sum(x_terms)
    def test_x(self):
        limit = len(self.test_db )-self.bit_n+1
        str_list = [self.test_db[i:i+self.bit_n] for i in range(limit)]
        xk_vect = [self.determine_x(str_list[i]) for i in range(limit)]
        return xk_vect


class Viterbi:
    def __init__(self,test_db,cluster_no,bits_n,prior_prob,transition_table,std_mean_cluster,test_x):
        self.calculated_answer = ""
        self.test_db = test_db
        self.cluster_no=cluster_no
        self.bits_n = bits_n
        self.prior_prob=prior_prob
        self.transition_table=transition_table
        self.std_mean_cluster=std_mean_cluster
        self.test_x=test_x
        limit = len(self.test_x)
        self.trellis =  [[0 for i in range(self.cluster_no)] for j in range(limit)]
        self.predecessor = [[0 for i in range(self.cluster_no)] for j in range(limit)]
        self.cluster_seq = []


    def gaussian(self,x,mean,variance_sqrt):
        return pow(math.sqrt(2*math.pi)*variance_sqrt,-1)*math.exp(-((x-mean)*(x-mean))/(2*variance_sqrt*variance_sqrt))


    def accuracy_viterbi(self):

        limit = len(self.test_db)
        acc_list = [self.test_db[i]==self.calculated_answer[i] for i in range(limit)]
        accuracy = acc_list.count(True)/limit
        print("Acc: ",accuracy* 100)



    def populate_first_row(self):
        x = self.test_x[0]
        gaussian_val = [self.gaussian(x,self.std_mean_cluster[i][0],self.std_mean_cluster[i][1]) for i in range(self.cluster_no)]
        self.trellis[0] = [self.prior_prob[i]*gaussian_val[i] for i in range(self.cluster_no)]


    def determine_singleNode(self,i,j):
                x_ = self.test_x[i]
                mean = self.std_mean_cluster[j][0]
                standard_dev = self.std_mean_cluster[j][1]
                gaussian_val = self.gaussian(x_,mean,standard_dev)


                tempList = [self.trellis[i-1][k] * self.transition_table[k][j] * gaussian_val for k in range(self.cluster_no)]
                self.trellis[i][j],self.predecessor[i][j] = max(tempList),tempList.index(max(tempList))

    def populate_trellis(self):
        self.populate_first_row()
        limit = len(self.test_x)
        [[self.determine_singleNode(i,j) for j in range(self.cluster_no)] for i in range(1,limit)]

    def printGraphs(self):
        print()
        print("TRELLIS GRAPH")
        [print(self.trellis[i]) for i in range(len(self.test_x))]
        print()
        print()

        print()
        print("Predecessor GRAPH")
        [print(self.predecessor[i]) for i in range(len(self.test_x))]
        print()
        print()

    def make_cluster_seq(self):
        t = self.trellis[len(self.test_x)-1]
        self.cluster_seq.append(t.index(max(t))) #las row maximum prob's index

        temp_index = self.cluster_seq[0]
        #print(self.cluster_seq)
        for i in range(len(self.test_x)-1,0,-1):
            self.cluster_seq.append(self.predecessor[i][temp_index])
            temp_index = self.predecessor[i][temp_index]
            #print(self.cluster_seq)

        self.cluster_seq.reverse()


    def calculate_string(self,binary_seq):
        self.calculated_answer+=binary_seq[0]
        bits = [binary_seq[i][ self.bits_n-1] for i in range(1,len(binary_seq))]
        self.calculated_answer += ''.join(bits)

    def print_answers(self,binary_seq):
        print("bin sequence: ",binary_seq)
        print("calculated string: ",self.calculated_answer)
        print("test DB          :",self.test_db)
        print()
        print()

    def viterbi_algo2(self):

        self.populate_trellis()
        self.printGraphs()

        self.make_cluster_seq()
        # bin(6)[2:].zfill(8)


        binary_seq = [bin(self.cluster_seq[i])[2:].zfill(self.bits_n) for i in range(len(self.cluster_seq))]

        self.calculate_string(binary_seq)

        self.print_answers(binary_seq)






def implementation():
    configuration = "config.txt"
    train_file = "train.txt"
    test_file = "test.txt"


    rd = ReadFiles(configuration,train_file,test_file)
    n,weights_array, number_clusters,variance,variance_sq =rd.configuration_input()
    train,test = rd.train_test_read()

    pr = Prior_determination(n,number_clusters,train)
    prior_probs = pr.prior_prob2()

    tr = Transition_states(train,n,number_clusters)
    transition_tab = tr.transition_table2()

    clstr = Clustering(number_clusters,train,n,variance_sq,weights_array)
    cluster_mean_std = clstr.std_mean_cluster()
    tst_x = Equalizer_Input(test,n,weights_array,variance)
    xK = tst_x.test_x()

    v = Viterbi(test,number_clusters,n,prior_probs,transition_tab,cluster_mean_std,xK)
    v.viterbi_algo2()
    v.accuracy_viterbi()

implementation()




