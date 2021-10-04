import os
import sys
import pandas as pd
import numpy as np
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.preprocessing import OneHotEncoder
# from scipy.special import softmax

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

def get_model(param_dict):
    weights_activation_list = []
    pass

def main(args):
    input_path, out_path, param_file = args[1:]
    param_dict = get_param_dict(param_file)

    X_train, Y_train, X_test = load_data(input_path)

    assert X_train.shape[0] == Y_train.shape[0]
    assert X_train.shape[1] == X_test.shape[1]







    

if __name__ == "__main__":
    args = sys.argv
    main(args)