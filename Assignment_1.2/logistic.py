import sys
import pandas as pd
import numpy as np
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax

from tqdm import tqdm
# TODO remove tqdm dependencies

# class_num_dict_tmp = [['Health Service Area', 8], ['Hospital County', 57], ['Facility Name', 212], ['Age Group', 5], ['Zip Code - 3 digits', 50], ['Gender', 3], ['Race', 4], ['Ethnicity', 4], ['Type of Admission', 6], ['Patient Disposition', 19], ['CCS Diagnosis Description', 260], ['CCS Procedure Description', 224], ['APR DRG Description', 308], ['APR MDC Description', 24], ['APR Severity of Illness Description', 4], ['APR Risk of Mortality', 4], ['APR Medical Surgical Description', 2], ['Payment Typology 1', 10], ['Payment Typology 2', 11], ['Payment Typology 3', 11], ['Emergency Department Indicator', 2]]
def load_data(train_file_name, test_file_name):
    train = pd.read_csv(train_file_name, index_col = 0)    
    test = pd.read_csv(test_file_name, index_col = 0)
        
    Y_train = np.array(train['Length of Stay'])
    # 1-8 encoding
    Y_train -= 1
    # 0-7 encoding

    # TODO: remove hardcoded 8
    one_hot_shape = (Y_train.shape[0], Y_train.max()+1)
    # one_hot_shape = (Y_train.shape[0], 8)

    one_hot = np.zeros(one_hot_shape)
    rows = np.arange(Y_train.shape[0])

    one_hot[rows, Y_train] = 1
    Y_train = one_hot

    train = train.drop(columns = ['Length of Stay'])

    #Ensuring consistency of One-Hot Encoding

    data = pd.concat([train, test], ignore_index = True)
    cols = train.columns
    cols = cols[:-1]
    data = pd.get_dummies(data, columns=cols, drop_first=True)
    data = data.to_numpy()
    X_train = data[:train.shape[0], :]
    X_test = data[train.shape[0]:, :]

    # Add a dummy ones in the data to account for bias
    X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis = 1)
    X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis = 1)

    return X_train, Y_train, X_test

def get_param_dict_a(param_file):

    return_dict = {"mode": None,
                    "n0": None,
                    "alpha":None,
                    "beta": None,
                    "n_iter": None}

    with open(param_file, "r") as f:
        lines = f.readlines()
        mode = int(lines[0])
        return_dict["mode"] = mode

        if(mode == 1):
            n0 = float(lines[1])
            return_dict["n0"] = n0
        elif(mode == 2):
            n0 = float(lines[1])
            return_dict["n0"] = n0
        elif(mode == 3):
            n0, alpha, beta  = list(map(float,lines[1].split(",")))
            return_dict["n0"] = n0
            return_dict["alpha"] = alpha
            return_dict["beta"] = beta

        n_iter = int(lines[2])
        return_dict["n_iter"] = n_iter

    print(return_dict)
    return return_dict

def get_param_dict_b(param_file):

    return_dict = {"mode": None,
                    "n0": None,
                    "alpha":None,
                    "beta": None,
                    "n_iter": None,
                    "bs": None}

    with open(param_file, "r") as f:
        lines = f.readlines()
        mode = int(lines[0])
        return_dict["mode"] = mode

        if(mode == 1):
            n0 = float(lines[1])
            return_dict["n0"] = n0
        elif(mode == 2):
            n0 = float(lines[1])
            return_dict["n0"] = n0
        elif(mode == 3):
            n0, alpha, beta  = list(map(float,lines[1].split(",")))
            return_dict["n0"] = n0
            return_dict["alpha"] = alpha
            return_dict["beta"] = beta

        n_iter = int(lines[2])
        bs = int(lines[3])
        return_dict["n_iter"] = n_iter
        return_dict["bs"] = bs

    print(return_dict)
    return return_dict

def get_lr(param_dict, t, W, X_train, Y_train, d, n0):
    # t is the iteration number of gradient update
    mode = param_dict["mode"]
    if(mode == 1):
        return param_dict["n0"]
    elif(mode == 2):
        return param_dict["n0"]/(t**0.5)
    elif(mode == 3):
        # dunno how to do this
        alpha = param_dict["alpha"]
        beta = param_dict["beta"]

        lr = n0
        # import pdb; pdb.set_trace()
        baseline_loss = loss_given_weight(X_train, Y_train, W)
        while(loss_given_weight(X_train, Y_train, W - lr*d) > baseline_loss - alpha*lr*np.linalg.norm(d)**2 ):
            lr*=beta

        return lr

def loss(Y_train, Y_hat_train):
    return -(np.log(Y_hat_train)*Y_train).sum()/(Y_train.shape[0])

def loss_given_weight(X_train, Y_train, W):
    gamma = 10**(-15)
    return -(np.log(np.clip(softmax(np.matmul(X_train, W), axis = 1), gamma, 1-gamma))*Y_train).sum()/(Y_train.shape[0])

        
def get_lr_for_part_c(param_dict, t, W, X_train, Y_train, d, n0):
    return 0.25
        


def mode_a(args):
    train_file_name, test_file_name, param_file, output_file_name, weight_file_name = args[2:] 
    X_train, Y_train, X_test = load_data(train_file_name, test_file_name)
    # Y_train shape is [batch,k]
    assert X_train.shape[1] == X_test.shape[1]

    param_dict = get_param_dict_a(param_file)
    n_iter = param_dict["n_iter"]
    lr = param_dict["n0"]
    n0 = param_dict["n0"]

    # initialise the weight matrix
    W = np.zeros((X_train.shape[1], 8))
    # W = np.random.rand(X_train.shape[1], 8)
    
    pbar = tqdm(range(1,n_iter+1))
    for t in pbar:
        Y_hat_train = softmax(np.matmul(X_train, W), axis = 1)
        # import pdb; pdb.set_trace()
        d = -np.matmul(X_train.T, (Y_train- Y_hat_train))/(X_train.shape[0])
        # import pdb; pdb.set_trace()
        lr = get_lr(param_dict,t, W, X_train, Y_train, d, n0)
        # print(Y_hat_train.argmax(axis = 1))

        W = W - lr*d

        pbar.set_postfix({"Loss": loss(Y_train, Y_hat_train), "LR": lr})
    
    Y_hat_test = softmax(np.matmul(X_test, W), axis = 1).argmax(axis = 1)
    # 0-7 encoding

    Y_hat_test+=1
    # 1-8 encoding

    with open(output_file_name, "w") as f:
        for y in Y_hat_test:
            f.write(str(y))
            f.write("\n")

    with open(weight_file_name, "w") as f:
        for w in W.reshape(-1):
            f.write(str(w))
            f.write("\n")

    # import pdb; pdb.set_trace()
    pass

def mode_b(args):
    train_file_name, test_file_name, param_file, output_file_name, weight_file_name = args[2:] 
    X_train, Y_train, X_test = load_data(train_file_name, test_file_name)
    # Y_train shape is [batch,k]
    assert X_train.shape[1] == X_test.shape[1]

    param_dict = get_param_dict_b(param_file)
    n_iter = param_dict["n_iter"]
    lr = param_dict["n0"]
    n0 = param_dict["n0"]
    bs = param_dict["bs"]

    # initialise the weight matrix
    W = np.zeros((X_train.shape[1], 8))
    # W = np.random.rand(X_train.shape[1], 8)
    
    pbar = tqdm(range(1,n_iter+1))
    for t in pbar:
        Y_hat_train = softmax(np.matmul(X_train, W), axis = 1)
        # import pdb; pdb.set_trace()
        d = -np.matmul(X_train.T, (Y_train- Y_hat_train))/(X_train.shape[0])
        # import pdb; pdb.set_trace()
        lr = get_lr(param_dict,t, W, X_train, Y_train, d, n0)
        for batch in range((X_train.shape[0]-1)//bs + 1):
            if(X_train[batch*bs:batch*bs + bs, :].shape[0]!= bs):
                # import pdb; pdb.set_trace()
                pass
            else:

                Y_hat_train = softmax(np.matmul(X_train[batch*bs:batch*bs + bs, :], W), axis = 1)
                # import pdb; pdb.set_trace()
                d = -np.matmul(X_train[batch*bs:batch*bs + bs, :].T, (Y_train[batch*bs:batch*bs + bs, :]- Y_hat_train))/(X_train[batch*bs:batch*bs + bs, :].shape[0])
                # import pdb; pdb.set_trace()
                # lr = get_lr(param_dict,t, W, X_train[batch*bs:batch*bs + bs, :], Y_train[batch*bs:batch*bs + bs, :], d, n0)
                # print(Y_hat_train.argmax(axis = 1))

                W = W - lr*d
                pbar.set_postfix({"Loss": loss(Y_train[batch*bs:batch*bs + bs, :], Y_hat_train), "LR": lr, "Iter": str(batch+1)+ "/" + str((X_train.shape[0]-1)//bs + 1) })

    
    Y_hat_test = softmax(np.matmul(X_test, W), axis = 1).argmax(axis = 1)
    # 0-7 encoding

    Y_hat_test+=1
    # 1-8 encoding

    with open(output_file_name, "w") as f:
        for y in Y_hat_test:
            f.write(str(y))
            f.write("\n")

    with open(weight_file_name, "w") as f:
        for w in W.reshape(-1):
            f.write(str(w))
            f.write("\n")

    # import pdb; pdb.set_trace()
    pass

def mode_c(args):
    train_file_name, test_file_name, param_file, output_file_name, weight_file_name = args[2:] 
    X_train, Y_train, X_test = load_data(train_file_name, test_file_name)
    # Y_train shape is [batch,k]
    assert X_train.shape[1] == X_test.shape[1]

    param_dict = get_param_dict_b(param_file)
    n_iter = 300
    lr = 0.25
    n0 = 0.25
    bs = 1000

    # TODO intermitten saving at 100 150 200 150 300 350 epoch

    # initialise the weight matrix
    W = np.zeros((X_train.shape[1], 8))
    # W = np.random.rand(X_train.shape[1], 8)
    
    pbar = tqdm(range(1,n_iter+1))
    for t in pbar:
        Y_hat_train = softmax(np.matmul(X_train, W), axis = 1)
        # import pdb; pdb.set_trace()
        d = -np.matmul(X_train.T, (Y_train- Y_hat_train))/(X_train.shape[0])
        # import pdb; pdb.set_trace()
        lr = get_lr_for_part_c(param_dict,t, W, X_train, Y_train, d, n0)
        for batch in range((X_train.shape[0]-1)//bs + 1):
            if(X_train[batch*bs:batch*bs + bs, :].shape[0]!= bs):
                # import pdb; pdb.set_trace()
                pass
            else:

                Y_hat_train = softmax(np.matmul(X_train[batch*bs:batch*bs + bs, :], W), axis = 1)
                # import pdb; pdb.set_trace()
                d = -np.matmul(X_train[batch*bs:batch*bs + bs, :].T, (Y_train[batch*bs:batch*bs + bs, :]- Y_hat_train))/(X_train[batch*bs:batch*bs + bs, :].shape[0])
                # import pdb; pdb.set_trace()
                # lr = get_lr(param_dict,t, W, X_train[batch*bs:batch*bs + bs, :], Y_train[batch*bs:batch*bs + bs, :], d, n0)
                # print(Y_hat_train.argmax(axis = 1))

                W = W - lr*d
                pbar.set_postfix({"Loss": loss(Y_train[batch*bs:batch*bs + bs, :], Y_hat_train), "LR": lr, "Iter": str(batch+1)+ "/" + str((X_train.shape[0]-1)//bs + 1) })

        if(t%50 == 0):
            Y_hat_test = softmax(np.matmul(X_test, W), axis = 1).argmax(axis = 1)
            # 0-7 encoding

            Y_hat_test+=1
            # 1-8 encoding

            with open(output_file_name, "w") as f:
                for y in Y_hat_test:
                    f.write(str(y))
                    f.write("\n")

            with open(weight_file_name, "w") as f:
                for w in W.reshape(-1):
                    f.write(str(w))
                    f.write("\n")
    

    # import pdb; pdb.set_trace()
    pass

def mode_d(args):
    pass



def main(args):
    mode = args[1]
    print("The mode is ", mode)
    if(mode == "a"):
        mode_a(args)
    if(mode == "b"):
        mode_b(args)
    if(mode == "c"):
        mode_c(args)
    if(mode == "d"):
        mode_d(args)

if __name__ == "__main__":
    args = sys.argv
    main(args)