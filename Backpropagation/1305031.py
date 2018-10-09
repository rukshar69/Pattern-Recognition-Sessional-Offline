import numpy as np

from matplotlib import pyplot as plt

class weight_bias:
    def __init__(self,p_layer_p):
        self.neurons_p_layer = p_layer_p
        self.mu = 0
        self.sigma = 1.5
        self.W = []
        self.B = []
        self.layerNum = len(p_layer_p)

    def init_weight(self):
        np.random.seed(10000)
        weights = []
        i = 0

        while (i< len(self.neurons_p_layer)-1 ):


            tempWeight = np.random.normal(self.mu, self.sigma,(self.neurons_p_layer[i + 1], self.neurons_p_layer[i]))

            weights.append(tempWeight)
            i += 1

        return weights

    def inti_bias(self):
        np.random.seed(10000)

        biases = []
        i = 0

        while (i< len(self.neurons_p_layer) -1):

            tempBias = np.random.normal(self.mu, self.sigma, (self.neurons_p_layer[i + 1], 1))
            i += 1

            biases.append(tempBias)
        return biases
    def getW_B(self):
        self.W =self.init_weight()
        self.B = self.inti_bias()
        return self.W, self.B

class Feed_Forward:
    def __init__(self,i, w, b):
        self.a = 0;
        self.i_ = i;
        self.w_ = w
        self.b_ = b
        self.neuron_output = []
        self.neuron_input = []
    def activation(self,x,option):
        if option == 1:
            t0 =  np.exp(-x)
            t1= (1 +t0)
            temp = 1 / t1
            return temp
        elif option == 2:
            a = self.activation(x,1)
            a= a * (1 - a)
            return a;
    def f_frwrd(self):

        input_output_pair = []
        i = 0
        feed_I = self.i_
        while ( i< len(self.w_)):
            input_into_neuron = np.matmul(self.w_[i],feed_I) +  self.b_[i]
            output_from_neuron = self.activation(input_into_neuron,1)
            input_output_pair.append([input_into_neuron,output_from_neuron])
            feed_I = output_from_neuron
            i+= 1

        i = 0
        self.neuron_output.append(self.i_)
        self.neuron_input.append(0)
        while ( i< len(self.b_)):
            temp = input_output_pair[i]
            self.neuron_input.append(temp[0])
            self.neuron_output.append(temp[1])
            i+= 1

        return feed_I, self.neuron_output, self.neuron_input

class backProp:

    def __init__(self,per_per_layer,epoch):

        self.del_W = 0
        self.del_B = 0
        self.cost_per_epoch = 0;
        self.no_layers = len(per_per_layer)
        self.per_per_layer = per_per_layer
        wb = weight_bias(self.per_per_layer)
        self.W,self.B =wb.getW_B()
        self.costList = []
        self.iterationList = []
        self.epoch = epoch

    def refresh(self):
        self.del_W = 0
        self.del_B = 0
        self.cost_per_epoch = 0;
    def activation(self,x,option):
        if option == 1:
            t0 =  np.exp(-x)
            t1= (1 +t0)
            temp = 1 / t1
            return temp
        elif option == 2:
            a = self.activation(x,1)
            a= a * (1 - a)
            return a;

    def determineCost(self,y_true, y_calculated):
        t0 = (y_true - y_calculated)
        t1 = t0**2
        t2 = .5*t1
        return t2


    def updDeltaB(self, deltas):
        if(self.del_B!=0):
            size = len(deltas)
            for i in range(size):
                self.del_B[i]=self.del_B[i]+deltas[i]

        elif self.del_B == 0:
            self.del_B = deltas

    def updDeltaW(self, derivations):
        if(self.del_W!=0):
            size = len(derivations)
            for i in range(size):
                self.del_W[i]=self.del_W[i]+derivations[i]

        elif self.del_W == 0:
            self.del_W = derivations


    def getDeltas(self,deltas,n_out, n_inp,y_true):
        size = len(deltas)
        dJdh =  -(y_true - n_out[size - 1])
        dhdz =  self.activation(n_inp[size - 1],2)
        deltas[size - 1] = dJdh *dhdz


        for i in range(size - 2, 0, -1):
            tranposeW = np.transpose(self.W[i])
            numpy_mult = np.matmul(tranposeW, deltas[i + 1])
            deltas[i] = numpy_mult * self.activation(n_inp[i],2)

        return deltas

    def det_derivative(self,deltas,n_out):
        size = len(deltas)
        derivations = []
        for i in range(1,size):

            derivations.append( np.matmul(deltas[i], np.transpose(n_out[i-1 ])))
        return derivations

    def updateDeltaWB(self,n_out, n_in,y_true):
        #####################################Gradient Descent###########################
        deltas = [0]*len(n_out)

        deltas = self.getDeltas(deltas,n_out,n_in,y_true)

        derivations = self.det_derivative(deltas,n_out)

        self.updDeltaB( deltas)
        self.updDeltaW(derivations)

    def updateDel(self,s,alpha,m):
        i = 0

        while (i< s):
            self.del_B[i+1] = alpha *1/m * self.del_B[i+1]
            self.del_W[i]= alpha *1/m * self.del_W[i]
            i += 1

    def updateWB(self,m):
        alpha = .5

        i = 0
        s = len(self.W)
        self.updateDel(s,alpha,m)
        while (i< s):
            self.W[i] =self.W[i] -  self.del_W[i]

            self.B[i] = self.B[i] -  self.del_B[i + 1]
            i += 1


    def separate_cost_iteration(self,cost_iteration_pair ):
        size = len(cost_iteration_pair)
        for i in range(size):
            temp = cost_iteration_pair[i]
            self.costList.append(temp[0])
            self.iterationList.append(temp[1])
    #def partial_deriv(self):
    def train(self,x, y_):
        cost_iteration_pair = []

        for i in range(self.epoch):
            #print("iteration number ",i)
            m  = len(x)
            self.refresh()
            for j in range(m):

                Fd = Feed_Forward(x[j], self.W, self.B)
                y, h, z =  Fd.f_frwrd()
                y_true = y_[j]
                y_calculated = y;
                self.updateDeltaWB(h,z,y_true )
                cost =self.determineCost(y_true, y_calculated )
                self.cost_per_epoch =self.cost_per_epoch + (sum(cost)/m)

            self.updateWB(m)

            iteration_number = i+1
            cost_iteration_pair.append([self.cost_per_epoch, iteration_number])

        self.separate_cost_iteration(cost_iteration_pair)
        return self.costList,self.iterationList,self.W, self.B



class graph_plot:
    def __init__(self,iterations_for_gr, cost_for_gr):
        self.iterations_for_gr=iterations_for_gr
        self.cost_for_gr=cost_for_gr

    def draw(self):
        plt.plot(self.iterations_for_gr,self.cost_for_gr)
        plt.xlabel("I")
        plt.ylabel("C")
        plt.show()

class IO:
    def __init__(self):
        self.inp_neuron = 0
        self.out_neuron = 0
        self.layers = 0
        self.neuron_per_layer = []

    def take_input(self):
        self.inp_neuron = int(input("number of input perceptrons: "))
        #print(type(self.inp_neuron))
        self.out_neuron = int(input("number of output perceptrons: "))
        self.layers = int(input("number of layers: "))
        #print "Let's talk about %s." % my_name
        print("There are %d hidden layer(s)"%(self.layers-2))
        hidden_layers = self.layers-2
        self.neuron_per_layer.append(self.inp_neuron)
        for i in range(hidden_layers):
            temp =  int(input("number of perceptrons for hidden layer %d: "%(i+1)))
            self.neuron_per_layer.append(temp)

        self.neuron_per_layer.append(self.out_neuron)
        return self.inp_neuron, self.out_neuron, self.neuron_per_layer

io_ = IO()
i_p_, o_p_, per_layer_p_ = io_.take_input()

class readFile:
    def __init__(self,train_file,test_file,input_perceptron, output_perceptron):
        self.tr_f = train_file
        self.tst_f = test_file
        self.mode = "r"
        #self.tr_f_open = open(self.tr_f,mode=mode)
        #self.tst_f_open = open(self.tst_f,mode=mode)
        self.inp_p = input_perceptron
        self.out_p = output_perceptron

        self.trainX = []
        self.testX = []

        self.trainY = []
        self.testY = []

    def read_train_features(self):
        tr_f_open = open(self.tr_f,mode=self.mode)
        for line in tr_f_open:
            line_numbers = line.split()
            features = []
            for j in range(self.inp_p):
                features.append(float(line_numbers[j]))
            self.trainX.append(features)

        tr_f_open.close()

    def read_train_classes(self):
        tr_f_open = open(self.tr_f,mode=self.mode)
        for line in tr_f_open:
            line_numbers = line.split()
            vectorized_classes = []
            for i in range(0,self.out_p):
                vectorized_classes.append(0)
            class_ = (line_numbers[self.inp_p])
            class_ = float(class_)
            class_ = int(class_)
            vectorized_classes[class_-1]=1
            self.trainY.append(vectorized_classes)

        tr_f_open.close()
        #self.print_array(self.trainY)

    def read_test_features(self):
        tst_f_open = open(self.tst_f,mode=self.mode)
        for line in tst_f_open:
            line_numbers = line.split()
            features = []
            for j in range(self.inp_p):
                features.append(float(line_numbers[j]))
            self.testX.append(features)

        tst_f_open.close()
        #self.print_array(self.testX)

    def read_test_classes(self):
        tst_f_open = open(self.tst_f,mode=self.mode)
        for line in tst_f_open:
            line_numbers = line.split()
            vectorized_classes = []
            for i in range(0,self.out_p):
                vectorized_classes.append(0)
            class_ = (line_numbers[self.inp_p])
            class_ = float(class_)
            class_ = int(class_)
            vectorized_classes[class_-1]=1
            self.testY.append(vectorized_classes)

        tst_f_open.close()
        #self.print_array(self.testY)

    def read_files(self):
        self.read_train_features()
        self.read_train_classes()
        self.read_test_features()
        self.read_test_classes()

    def modify_arrays(self):

        self.trainX = self.modify(self.trainX)
        self.trainY = self.modify(self.trainY)
        self.testX = self.modify(self.testX)
        self.testY = self.modify(self.testY)
        #self.print_array(self.trainY)

    def read_train_test(self):
        self.read_files()
        self.modify_arrays()

        return  self.trainX, self.trainY, self.testX,self.testY

    def transpose(self,m):
        temp = [m]
        temp = np.array(temp)
        temp = np.transpose(temp)
        return temp
    def modify(self,arr):
        tempArr = []
        length = len(arr)
        for i in range(length):
            line = arr[i]
            line = self.transpose(line)
            tempArr.append(line)
        tempArr = np.array(tempArr)

        return  tempArr
    def print_array(self,arr):
        for r in arr:
            print(r)
        print(len(arr))

def test_NN(trainedW8,traineddBias,tsX,tsY):
    count = 0
    x = tsX
    y = tsY
    for i in range(0, len(x)):
        Fd = Feed_Forward(x[i], trainedW8, traineddBias)
        test, h, z =  Fd.f_frwrd()
        y_predicted_index = np.argmax(test)
        y_actual_index = np.argmax(y[i])
        if (y_predicted_index == y_actual_index):
            count = count + 1
    return count / len(x)

def implement(i_p, o_p, per_layer_p,train_file_name, test_file_name):
    rd = readFile(train_file_name,test_file_name,i_p, o_p)
    trX, trY, tsX,tsY = rd.read_train_test()

    it = 100
    nn = backProp(per_layer_p,it)

    c,itera,trainedW8,traineddBias = nn.train(trX,trY) #cost , iteration
    tstNN = test_NN(trainedW8,traineddBias,tsX,tsY)
    return tstNN, c, itera

acc, cost_nn, iterations_nn = implement(i_p_, o_p_, per_layer_p_,"train.txt","test.txt")
print("Acc:",acc*100)
g = graph_plot(iterations_nn,cost_nn)
g.draw()














