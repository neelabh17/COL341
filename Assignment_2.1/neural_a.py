import os
import sys
import pandas as pd
import numpy as np
import math
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax

# from tqdm import tqdm

# class_num_dict_tmp = [['Health Service Area', 8], ['Hospital County', 57], ['Facility Name', 212], ['Age Group', 5], ['Zip Code - 3 digits', 50], ['Gender', 3], ['Race', 4], ['Ethnicity', 4], ['Type of Admission', 6], ['Patient Disposition', 19], ['CCS Diagnosis Description', 260], ['CCS Procedure Description', 224], ['APR DRG Description', 308], ['APR MDC Description', 24], ['APR Severity of Illness Description', 4], ['APR Risk of Mortality', 4], ['APR Medical Surgical Description', 2], ['Payment Typology 1', 10], ['Payment Typology 2', 11], ['Payment Typology 3', 11], ['Emergency Department Indicator', 2]]
def load_data(input_file):
    train_file_name = "toy_dataset_train.csv"
    train_file = os.path.join(input_file, train_file_name)

    test_file_name = "toy_dataset_test.csv"
    test_file = os.path.join(input_file, test_file_name)

    train = pd.read_csv(train_file)    
    # removes does not remove first column 
    test = pd.read_csv(test_file, index_col = 0)

    train = np.array(train)  
    Y_train = train[:,0]

    X_train = train[:, 1:]
    X_test = np.array(test)

    # import pdb; pdb.set_trace()

    return X_train, Y_train, X_test
def get_param_dict(param_file):

    return_dict = {"epoch": None,
                    "bs": None,
                    "arc":None,
                    "lr_mode": None,
                    "lr": None,
                    "f_act": None,
                    "loss": None,
                    "seed": None,
    }

    with open(param_file, "r") as f:
        lines = f.readlines()
        epoch = int(lines[0])
        bs = int(lines[1])
        arc = eval(lines[2])
        lr_mode = int(lines[3])
        lr = float(lines[4])
        f_act = int(lines[5])
        loss = int(lines[6])
        seed = int(lines[7])


        return_dict["epoch"] = epoch
        return_dict["bs"] = bs
        return_dict["arc"] = arc
        return_dict["lr_mode"] = lr_mode
        return_dict["lr"] = lr
        return_dict["f_act"] = f_act
        return_dict["loss"] = loss
        return_dict["seed"] = seed
        

    print(return_dict)
    return return_dict

def get_lr_func(mode):
    def const_lr(lr, **kwargs):
        return lr
    def adaptive_lr(lr, epoch, **kwargs):
        return lr/((epoch)**0.5)

    if(mode == 0):
        return const_lr
    else:
        return adaptive_lr

def get_act_func(mode):
    def log_sigmoid(A):
        # return the activation and the cache derivative
        f = 1/(1+ np.exp(-A))
        return np.log(f), 1-f
    def tanh(A):
        # return the activation and the cache derivative
        f = np.tanh(A)
        return f, 1-np.square(f)
    def relu(A):
        # return the activation and the cache derivative
        B = np.array(A)
        C = np.array(A)
        B[B<0] = 0
        C[C<0] = 0
        C[C==0] = 0
        C[C>0] = 1

        return B, C
    def softmax(A):
        # return the activation and the cache derivative
        exp = np.exp(A)
        sum = exp.sum(axis = 1, keepdims = True)
        return exp/sum, None

    if(mode == 0):
        return log_sigmoid
    elif(mode == 1):
        return tanh
    elif(mode == 2):
        return relu
    elif(mode == 3):
        return softmax

def get_derivative_func(mode):
    def der_log_sigmoid(A):
        return np.log(1/(1+ np.exp(-A)))
    def der_tanh(A):
        return np.tanh(A)
    def der_relu(A):
        B = np.array(A)
        B[B<0] = 0
        return B

    if(mode == 0):
        return der_log_sigmoid
    elif(mode == 1):
        return der_tanh
    else:
        return der_relu

class NNet:
    def __init__(self, input_size, param_dict, output_act_mode = 3):
        self.input_size = input_size

        self.weights = []
        self.activation_f = []
        self.activation_mode = []
        self.cache = []

        self.seed = param_dict["seed"]
        self.arc = param_dict["arc"]
        self.intermediate_activation_mode = param_dict["f_act"]
        self.bs = param_dict["bs"]
        self.epoch = param_dict["epoch"]

        self.final_activation_mode = output_act_mode 

        self.initialise_weights()
    
    def initialise_weights(self):
        assert len(self.weights) == 0
        assert len(self.activation_f) == 0
        assert len(self.cache) == 0
        assert len(self.activation_mode) == 0
        np.random.seed(self.seed)

        arc_list = [self.input_size] + self.arc

        for inp, out in zip(arc_list[:-1], arc_list[1:]):
            self.weights.append((np.random.normal(0, 1, size = (inp+1, out))*math.sqrt(2/(inp+31+out))).astype(np.float32()))
            self.activation_f.append(get_act_func(self.intermediate_activation_mode))
            self.activation_mode.append(self.intermediate_activation_mode)

        # we have aded an extra activation in the last layer 
        # that needs to be changed to required activation

        # Changes for the final layer are made
        self.activation_f[-1] = get_act_func(self.final_activation_mode)
        self.activation_mode[-1] = self.final_activation_mode

        self.cache = [ None for _ in range(len(self.activation_mode)) ]

        print("INFO: Weights Initialized")


    def train(self, X_train, Y_train):
        bs = self.bs
        epoch = self.epoch
        for t in range(1,epoch + 1):
            for batch in range((X_train.shape[0]-1)//bs + 1):
                if(X_train[batch*bs:batch*bs + bs, :].shape[0]!= bs):
                    # import pdb; pdb.set_trace()
                    pass
                else:
                    Y_hat_train = self.forward(X_train[batch*bs:batch*bs + bs, :])
                    self.backward(Y_hat_train)

    def bias_transform(self, X):
        # x = [b, n_feat]
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        # print(X.shape)
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)

    def forward(self, X_train):
        X = X_train  
        for i , (weight, activation_func, activation_mode) in enumerate(zip(self.weights, self.activation_f, self.activation_mode)):
            X = self.bias_transform(X)
            X = np.matmul(X, weight)
            X, derivative = activation_func(X)
            self.cache[i] = derivative

    def backward(self, Y_hat_train):
        pass


def main(args):
    input_path, out_path, param_file = args[1:]
    param_dict = get_param_dict(param_file)

    X_train, Y_train, X_test = load_data(input_path)

    assert X_train.shape[0] == Y_train.shape[0]
    assert X_train.shape[1] == X_test.shape[1]

    model = NNet(input_size= 200, param_dict=param_dict)
    model.train(X_train, Y_train)







    

if __name__ == "__main__":
    args = sys.argv
    main(args)