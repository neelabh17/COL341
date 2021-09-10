import sys
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
class_num_dict_tmp = [['Health Service Area', 8], ['Hospital County', 57], ['Facility Name', 212], ['Age Group', 5], ['Zip Code - 3 digits', 50], ['Gender', 3], ['Race', 4], ['Ethnicity', 4], ['Type of Admission', 6], ['Patient Disposition', 19], ['CCS Diagnosis Description', 260], ['CCS Procedure Description', 224], ['APR DRG Description', 308], ['APR MDC Description', 24], ['APR Severity of Illness Description', 4], ['APR Risk of Mortality', 4], ['APR Medical Surgical Description', 2], ['Payment Typology 1', 10], ['Payment Typology 2', 11], ['Payment Typology 3', 11], ['Emergency Department Indicator', 2]]
class_num_dict = {}
for a,b in class_num_dict_tmp:
    class_num_dict[a] = b
def transforms(X, Y, trans, reg, enc, do_reg = False, df_x = None, df_y = None):
    # polynimial fitting
    X = trans.fit_transform(X)

    # one hot encoding
    for cls_name in tqdm(enc):
        print("\nOne hoting for ", cls_name)
        one_hot = enc[cls_name].transform(df_x[cls_name].to_numpy().reshape(-1,1)).toarray()
        # import pdb; pdb.set_trace()
        X = np.concatenate((X, one_hot ), axis = 1)
    
    print("After One hot")
    print(X.shape)

        

    if(do_reg):
        reg.fit(X, Y)


    X = X.T[reg.coef_ != 0].T
    dummy = np.array([1 for _ in range(X.shape[0])]).reshape(-1,1)
    X = np.concatenate((X,dummy), axis = 1)
    return X

# import the data
def main(args):
    data_file = args[1]
    data = pd.read_csv(data_file)
    # data.drop("Total Costs", inplace = True, axis = 1)
    data.drop("Unnamed: 0", inplace = True, axis = 1)
    data.drop("Facility Id", inplace = True, axis = 1)
    data.drop("CCS Diagnosis Code", inplace = True, axis = 1)
    data.drop("CCS Procedure Code", inplace = True, axis = 1)
    data.drop("APR DRG Code", inplace = True, axis = 1)
    data.drop("APR MDC Code", inplace = True, axis = 1)
    data.drop("APR Severity of Illness Code", inplace = True, axis = 1)

    # data["Birth Weight"] =(data["Birth Weight"] - data["Birth Weight"].mean()) / data["Birth Weight"].std()
    # data["Length of Stay"] =(data["Length of Stay"] - data["Length of Stay"].mean()) / data["Length of Stay"].std()
    mean = data["Total Costs"].mean()
    std = data["Total Costs"].std()
    print("Mean and std are ", mean, std)
    # data["Total Costs"] =(data["Total Costs"] - data["Total Costs"].mean()) / data["Total Costs"].std()
    enc = {}
    for class_name in class_num_dict:
        enc[class_name] = OneHotEncoder(handle_unknown='ignore')
        enc[class_name].fit(data[class_name].to_numpy().reshape(-1, 1))
    # import pdb; pdb.set_trace()




    train_val=data.sample(frac=0.8,random_state=200) #random state is a seed value
    test=data.drop(train_val.index)
    train = train_val.sample(frac=0.8,random_state=200)
    val = train_val.drop(train.index)


    train_val_y = train_val["Total Costs"]
    train_val.drop("Total Costs", inplace = True, axis = 1)
    train_val_x = train_val


    test_y = test["Total Costs"]
    test.drop("Total Costs", inplace = True, axis = 1)
    test_x = test

    # data_Y = data["Total Costs"]
    # data_X = data
    print(train.shape, val.shape, test.shape)

    # Super basic baseline


    lambdas = np.genfromtxt("regularization.txt", delimiter = ",")
    print(lambdas)
    best_r = []
    best_ls = []
    # Lasso LARS
    for lam in tqdm(lambdas):
        Y = train_val_y.to_numpy()
        X = train_val_x.to_numpy()
        # print()
        # print(X.shape,Y.shape)
        reg = linear_model.LassoLars(alpha=lam)
        # reg.fit(X, Y)


        # normal
        # W = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))

        # polynomial
        trans = PolynomialFeatures(degree=2)
        X = transforms(X, Y, trans, reg, enc, do_reg= True, df_x=train_val_x)
        # X_new = trans.fit_transform(X)
        # X = X_new
        # reg.fit(X_new, Y)
        # X = X_new.T[reg.coef_ != 0].T
        # # import pdb; pdb.set_trace()


        # dummy = np.array([1 for _ in range(X.shape[0])]).reshape(-1,1)
        # X = np.concatenate((X,dummy), axis = 1)
        W = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))

        print("W shape ", W.shape)


        # import pdb; pdb.set_trace()


        Y = test_y.to_numpy()
        X = test_x.to_numpy()
        X = transforms(X, Y, trans, reg, enc, do_reg= False, df_x = test_x)
        # X_new = trans.fit_transform(X)
        # X = X_new.T[reg.coef_ != 0].T
        # dummy = np.array([1 for _ in range(X.shape[0])]).reshape(-1,1)
        # X = np.concatenate((X,dummy), axis = 1)
        # # import pdb; pdb.set_trace()

        best_r.append([lam, 1 - np.linalg.norm(np.matmul(X, W)- Y)/ np.linalg.norm(Y), reg.score(X,Y), X.shape[1]])
        best_ls.append([lam, np.linalg.norm(np.matmul(X, W)- Y),X.shape[1]])

    # import pdb; pdb.set_trace()
    best_r.sort(key = lambda x:x [1], reverse = True)
    best_ls.sort(key = lambda x:x [1])
    print(np.array(best_r))
    print(np.array(best_ls))

    

if __name__ == "__main__":
    args = sys.argv
    main(args)