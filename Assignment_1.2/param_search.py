import sys
import pandas as pd
import numpy as np
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns


# from tqdm import tqdm

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
    # import pdb;pdb.set_trace()
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
    original_shape = X_train.shape[0]
    X_val = X_train[int(0.9*original_shape):, :]
    Y_val = Y_train[int(0.9*original_shape):, :]
    X_train = X_train[:int(0.9*original_shape), :]
    Y_train = Y_train[:int(0.9*original_shape), :]
    return X_train, Y_train, X_val, Y_val, X_test


def loss(Y_train, Y_hat_train):
    return -(np.log(Y_hat_train)*Y_train).sum()/(Y_train.shape[0])

def loss_given_weight(X_train, Y_train, W):
    gamma = 10**(-15)
    return -(np.log(np.clip(softmax(np.matmul(X_train, W), axis = 1), gamma, 1-gamma))*Y_train).sum()/(Y_train.shape[0])

 

def main(args):
    train_file_name, test_file_name, = args[1:] 
    X_train, Y_train, X_val, Y_val, X_test = load_data(train_file_name, test_file_name)
    # Y_train shape is [batch,k]
    assert X_train.shape[1] == X_test.shape[1]

    # param_dict = get_param_dict_b(param_file)

    # ------------ LR Search Started ------------------
    do_lr = False
    if(do_lr):
        n_iter = 100
        lrs = [2.5 * 10**(-i) for i in range(6)]    
        bs = 400

        epoch = [[] for lr in lrs]
        loss_train = [[] for lr in lrs]
        loss_val = [[] for lr in lrs]
        for i, lr in enumerate(lrs):
            print(i, "of", len(lrs))

            # initialise the weight matrix
            W = np.zeros((X_train.shape[1], 8))
            # W = np.random.rand(X_train.shape[1], 8)
            
            # pbar = tqdm(range(1,n_iter+1))
            # for t in pbar:
            for t in range(1,n_iter+1):
                Y_hat_train = softmax(np.matmul(X_train, W), axis = 1)
                # import pdb; pdb.set_trace()
                d = -np.matmul(X_train.T, (Y_train- Y_hat_train))/(X_train.shape[0])
                # import pdb; pdb.set_trace()
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
                        # pbar.set_postfix({"Loss": loss(Y_train[batch*bs:batch*bs + bs, :], Y_hat_train), "LR": lr, "Iter": str(batch+1)+ "/" + str((X_train.shape[0]-1)//bs + 1) })
                epoch[i].append(t)
                loss_train[i].append(loss_given_weight(X_train, Y_train, W))
                loss_val[i].append(loss_given_weight(X_val, Y_val, W))
        sns.set()
        for i, lr in enumerate(lrs):
            plt.plot(epoch[i], loss_train[i], label = "{:.1e}".format(lr))
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig("train_lr_changes_bs_same.jpg")

        plt.clf()
        for i, lr in enumerate(lrs):
            plt.plot(epoch[i], loss_val[i], label = "{:.1e}".format(lr))
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig("val_lr_changes_bs_same.jpg")

    # ------------ LR Search Ended ------------------




    # ------------ BS Search Started ------------------
    do_bs = True
    # import pdb; pdb.set_trace
    if(do_bs):
        n_iter = 100
        # bss = [90000]    
        # bss = [2**i for i in range(12)] + [90000]    
        bss = [10, 100, 1000, 10000, 90000]    
        lr = 0.25

        epoch = [[] for lr in bss]
        loss_train = [[] for lr in bss]
        loss_val = [[] for lr in bss]
        for i, bs in enumerate(bss):
            print(i, "of", len(bss))

            # initialise the weight matrix
            W = np.zeros((X_train.shape[1], 8))
            # W = np.random.rand(X_train.shape[1], 8)
            
            # pbar = tqdm(range(1,n_iter+1))
            # for t in pbar:
            for t in range(1,n_iter+1):
                Y_hat_train = softmax(np.matmul(X_train, W), axis = 1)
                # import pdb; pdb.set_trace()
                d = -np.matmul(X_train.T, (Y_train- Y_hat_train))/(X_train.shape[0])
                # import pdb; pdb.set_trace()
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
                        # pbar.set_postfix({"Loss": loss(Y_train[batch*bs:batch*bs + bs, :], Y_hat_train), "LR": lr, "Iter": str(batch+1)+ "/" + str((X_train.shape[0]-1)//bs + 1) })
                epoch[i].append(t)
                loss_train[i].append(loss_given_weight(X_train, Y_train, W))
                loss_val[i].append(loss_given_weight(X_val, Y_val, W))
        sns.set()
        for i, bs in enumerate(bss):
            plt.plot(epoch[i], loss_train[i], label = str(bs))
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig("train_bs_changes_lr_same.jpg")

        plt.clf()
        for i, bs in enumerate(bss):
            plt.plot(epoch[i], loss_val[i], label = str(bs))
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.savefig("val_bs_changes_lr_same.jpg")




if __name__ == "__main__":
    args = sys.argv
    main(args)