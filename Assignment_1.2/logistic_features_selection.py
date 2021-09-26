import sys
import pandas as pd
import numpy as np
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2


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
    print(train_file_name, test_file_name)
    X_train, Y_train, X_val, Y_val, X_test = load_data(train_file_name, test_file_name)
    # Y_train shape is [batch,k]
    assert X_train.shape[1] == X_test.shape[1]
    # import pdb; pdb.set_trace()
    model = SelectKBest(chi2, k=500)
    X_new = model.fit_transform(X_train, Y_train.argmax(axis = 1))
    print(model.get_support(True).tolist())
    # import pdb; pdb.set_trace()


    


if __name__ == "__main__":
    args = sys.argv
    main(args)