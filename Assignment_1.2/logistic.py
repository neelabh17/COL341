import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder

# class_num_dict_tmp = [['Health Service Area', 8], ['Hospital County', 57], ['Facility Name', 212], ['Age Group', 5], ['Zip Code - 3 digits', 50], ['Gender', 3], ['Race', 4], ['Ethnicity', 4], ['Type of Admission', 6], ['Patient Disposition', 19], ['CCS Diagnosis Description', 260], ['CCS Procedure Description', 224], ['APR DRG Description', 308], ['APR MDC Description', 24], ['APR Severity of Illness Description', 4], ['APR Risk of Mortality', 4], ['APR Medical Surgical Description', 2], ['Payment Typology 1', 10], ['Payment Typology 2', 11], ['Payment Typology 3', 11], ['Emergency Department Indicator', 2]]
def load_data(train_file_name, test_file_name):
    train = pd.read_csv(train_file_name, index_col = 0)    
    test = pd.read_csv(test_file_name, index_col = 0)
        
    Y_train = np.array(train['Length of Stay'])

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
    X_train = np.concatenate((X_train, np.ones((X_train.shape[0], 1))), axis = 1)
    X_test = np.concatenate((X_test, np.ones((X_test.shape[0], 1))), axis = 1)

    return X_train, Y_train, X_test

def get_param_dict(param_file):

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
        

        


def mode_a(args):
    train_file_name, test_file_name, param_file, output_file_name, weight_file_name = args[2:] 
    X_train, Y_train, X_test = load_data(train_file_name, test_file_name)
    assert X_train.shape[1] == X_test.shape[1]

    param_dict = get_param_dict(param_file)

    import pdb; pdb.set_trace()
    # initialise the weight matrix
    W = np.zeros(X_train.shape[1], 8)





    d = None
    lr = None

    W = W - lr*d


    pass

def mode_b(args):
    pass

def mode_c(args):
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