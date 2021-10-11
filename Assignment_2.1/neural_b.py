import os
import sys
import pandas as pd
import numpy as np
import math
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
# TODO remove tqdm dependencies
# class_num_dict_tmp = [['Health Service Area', 8], ['Hospital County', 57], ['Facility Name', 212], ['Age Group', 5], ['Zip Code - 3 digits', 50], ['Gender', 3], ['Race', 4], ['Ethnicity', 4], ['Type of Admission', 6], ['Patient Disposition', 19], ['CCS Diagnosis Description', 260], ['CCS Procedure Description', 224], ['APR DRG Description', 308], ['APR MDC Description', 24], ['APR Severity of Illness Description', 4], ['APR Risk of Mortality', 4], ['APR Medical Surgical Description', 2], ['Payment Typology 1', 10], ['Payment Typology 2', 11], ['Payment Typology 3', 11], ['Emergency Department Indicator', 2]]
def load_data(input_file):
    train_file_name = "train_data_shuffled.csv"
    train_file = os.path.join(input_file, train_file_name)

    test_file_name = "public_test.csv"
    test_file = os.path.join(input_file, test_file_name)

    # import pdb; pdb.set_trace()
    train = pd.read_csv(train_file, header = None)    
    test = pd.read_csv(test_file, header = None)

    train = np.array(train)  
    Y_train = train[:,-1]
    X_train = train[:, :-1]
    
    test = np.array(test)  
    Y_test = test[:,-1]
    X_test = test[:, :-1]



    return X_train/255, Y_train, X_test/255, Y_test
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
        # 0 for CE 1 for MSE
        seed = int(lines[7])


        return_dict["epoch"] = epoch
        return_dict["bs"] = bs
        return_dict["arc"] = arc
        return_dict["lr_mode"] = lr_mode
        return_dict["lr"] = lr
        return_dict["f_act"] = f_act
        return_dict["loss"] = loss
        return_dict["seed"] = seed
        

    # print(return_dict)
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
        der = f*(1-f)
        # der = np.concatenate((der, np.zeros((der.shape[0], 1))), axis = 1)

        return f, der
    def tanh(A):
        # return the activation and the cache derivative
        f = np.tanh(A)
        der = 1-np.square(f) 
        # der = np.concatenate((der, np.zeros((der.shape[0], 1))), axis = 1)

        return f, der
    def relu(A):
        # return the activation and the cache derivative
        B = np.array(A)
        C = np.array(A)
        B[B<0] = 0
        C[C<0] = 0
        C[C==0] = 0
        C[C>0] = 1
        der = C
        # der = np.concatenate((der, np.zeros((der.shape[0], 1))), axis = 1)
        return B, der
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

class NNet:
    def __init__(self, input_size, param_dict, out_path):
        self.input_size = input_size
        self.out_path = out_path

        self.weights = None
        self.activation_f = None
        self.activation_mode = None
        self.cache = None
        self.Z = None
        self.dA = None
        self.current_epoch = None
        

        self.seed = param_dict["seed"]
        self.arc = param_dict["arc"]
        self.intermediate_activation_mode = param_dict["f_act"]
        self.bs = param_dict["bs"]
        self.epoch = param_dict["epoch"]
        self.loss_mode = param_dict["loss"]
        self.lr_mode = param_dict["lr_mode"]
        self.lr = param_dict["lr"]
        self.num_classes = self.arc[-1]

        if(self.loss_mode == 0):
            self.final_activation_mode = 3
        else:
            self.final_activation_mode = self.intermediate_activation_mode


        self.initialise_weights()
    def save_weights(self):
        # print("INFO: Saving weights")
        for i, weight in enumerate(self.weights[1:]):
            np.save(os.path.join(self.out_path,"w_{}.npy".format(i+1)),weight)
    def save_weights_iter(self):
        # print("INFO: Saving weights")
        for i, weight in enumerate(self.weights[1:]):
            np.save(os.path.join(self.out_path,"w_{}_iter.npy".format(i+1)),weight)
    
    def initialise_weights(self):
        assert self.weights == None
        assert self.activation_f == None
        assert self.cache == None
        assert self.activation_mode == None
        assert self.Z == None
        assert self.dA == None
        np.random.seed(self.seed)



        arc_list = [self.input_size] + self.arc
        
        self.weights = [ None for _ in range(len(arc_list)) ]
        self.activation_f = [ None for _ in range(len(arc_list)) ]
        self.activation_mode = [ None for _ in range(len(arc_list)) ]
        self.cache = [ None for _ in range(len(arc_list)) ]
        self.Z = [ None for _ in range(len(arc_list)) ]
        self.dA = [ None for _ in range(len(arc_list)) ]

        for i, (inp, out) in enumerate(zip(arc_list[:-1], arc_list[1:])):
            # import pdb;pdb.set_trace()
            # self.weights[i + 1] = np.float32(np.random.normal(0, 1, size = (inp+1, out))*math.sqrt(2/(inp+1+out)))
            self.weights[i + 1] = np.float64((np.random.normal(0, 1, size = (inp+1, out))*math.sqrt(2/(inp+1+out))).astype(np.float32()))
            # self.weights[i + 1] = (np.random.normal(0, 1, size = (inp+1, out))*math.sqrt(2/(inp+1+out))).astype(np.float32())
            self.activation_f[i + 1] = get_act_func(self.intermediate_activation_mode)
            self.activation_mode[i + 1] = self.intermediate_activation_mode

        # we have aded an extra activation in the last layer 
        # that needs to be changed to required activation

        # Changes for the final layer are made
        self.activation_f[-1] = get_act_func(self.final_activation_mode)
        self.activation_mode[-1] = self.final_activation_mode


        # print("INFO: Weights Initialized")


    def train(self, X_train, Y_train):
        bs = self.bs
        self.current_epoch = 1
        # pbar = tqdm(total=self.epoch)
        while(self.current_epoch <= self.epoch):
        # for t in range(1,epoch + 1):
            for batch in range((X_train.shape[0]-1)//bs + 1):
                if(X_train[batch*bs:batch*bs + bs, :].shape[0]!= bs):
                    # import pdb; pdb.set_trace()
                    pass
                else:
                    self.forward(X_train[batch*bs:batch*bs + bs, :])
                    # import pdb; pdb.set_trace()
                    self.backward(Y_train[batch*bs:batch*bs + bs, :])
                    if(batch + 1  == 5 and self.current_epoch == 1):
                        self.save_weights_iter()

            
            # if(self.current_epoch == 5):
            #     self.save_weights()
            self.current_epoch += 1
            # pbar.update()

    def bias_transform(self, X):
        # x = [b, n_feat]
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        # print(X.shape)
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1)

    def forward(self, X_train):
        X = X_train  
        for i , (weight, activation_func, activation_mode) in enumerate(zip(self.weights[1:], self.activation_f[1:], self.activation_mode[1:])):
            X = self.bias_transform(X)
            self.Z[i] = X
            # import pdb; pdb.set_trace()
            X = np.dot(X, weight)
            X, derivative = activation_func(X)
            # self.Z[i + 1] = X
            self.cache[i + 1] = derivative
        
        # might be wrong
        self.Z[-1] = X

    def backward(self, Y_train):
        # Assert that CE loss is used with softmax activation
        if(self.loss_mode == 0):
            assert self.final_activation_mode == 3
            self.dA[-1] = (self.Z[-1] - Y_train)/Y_train.shape[0]

        # Assert that MSE loss is NOT used with softmax activation because it makes things trickier
        elif(self.loss_mode == 1):
            assert self.final_activation_mode != 3
            self.dA[-1] = ((self.Z[-1] - Y_train)/Y_train.shape[0])*self.cache[-1]

        l = len(self.dA) - 2
        while(l>= 1):
            # import pdb; pdb.set_trace()
            self.dA[l] = np.dot(self.dA[l+1], self.weights[l+1].T)[:,1:]*self.cache[l]

            l-=1
        
        l = len(self.dA) - 1
        while(l>= 1):
            dw_l  = np.dot(self.Z[l-1].T, self.dA[l])
            lr = self.get_lr()
            self.weights[l] -= lr*dw_l 
            l-=1
        pass


    def get_lr(self):
        if(self.lr_mode == 0):
            return self.lr
        else:
            return self.lr/(math.sqrt(self.current_epoch))

    def eval(self, X_test):
        X = X_test  
        for _ , (weight, activation_func, activation_mode) in enumerate(zip(self.weights[1:], self.activation_f[1:], self.activation_mode[1:])):
            X = self.bias_transform(X)
            # import pdb; pdb.set_trace()
            X = np.dot(X, weight)
            X, _ = activation_func(X)
            # self.Z[i + 1] = X

        return X.argmax(axis = 1)


            
def one_hot(X, num_classes):
    # X shape : (n, )
    # import pdb; pdb.set_trace()
    Y = np.zeros((X.shape[0], num_classes))
    rows = np.arange(X.shape[0])
    Y[rows, X.tolist()] = 1

    return Y

def main(args):
    input_path, out_path, param_file = args[1:]
    param_dict = get_param_dict(param_file)

    X_train, Y_train, X_test, Y_test = load_data(input_path)

    assert X_train.shape[0] == Y_train.shape[0]
    assert X_train.shape[1] == X_test.shape[1]

    model = NNet(input_size = 1024, param_dict=param_dict, out_path=out_path)
    num_classes = model.num_classes

    Y_train = one_hot(Y_train, num_classes)
    model.train(X_train, Y_train)
    model.save_weights()

    Y_pred = model.eval(X_test)
    np.save(os.path.join(out_path,"predictions.npy"), Y_pred)
     

if __name__ == "__main__":
    args = sys.argv
    main(args)