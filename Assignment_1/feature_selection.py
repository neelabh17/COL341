import sys
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm

def transforms(X):
    pass

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

    data["Birth Weight"] =(data["Birth Weight"] - data["Birth Weight"].mean()) / data["Birth Weight"].std()
    data["Length of Stay"] =(data["Length of Stay"] - data["Length of Stay"].mean()) / data["Length of Stay"].std()


    train_val=data.sample(frac=0.8,random_state=200) #random state is a seed value
    test=data.drop(train_val.index)
    train = train_val.sample(frac=0.8,random_state=200)
    val = train_val.drop(train.index)
    train_val_y = train_val["Total Costs"].to_numpy()
    train_val.drop("Total Costs", inplace = True, axis = 1)
    train_val_x = train_val

    # data_Y = data["Total Costs"]
    # data_X = data
    print(train.shape, val.shape, test.shape)


    # Super basic baseline


    lambdas = np.genfromtxt("regularization.txt", delimiter = ",")
    print(lambdas)
    # Lasso LARS
    for lam in tqdm(lambdas):
        Y = train_val_y
        X = train_val_x
        print(X,Y.shape)
        reg = linear_model.LassoLars(alpha=lam)
        # reg.fit(X, Y)


        # normal
        # W = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))

        # polynomial
        trans = PolynomialFeatures(degree=2)
        X_new = trans.fit_transform(X)
        X = X_new
        reg.fit(X_new, Y)
        X = X_new.T[reg.coef_ != 0].T
        # import pdb; pdb.set_trace()


        dummy = np.array([1 for _ in range(X.shape[0])]).reshape(-1,1)
        X = np.concatenate((X,dummy), axis = 1)
        W = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))


        # import pdb; pdb.set_trace()


        Y = test["Total Costs"].to_numpy()
        test.drop("Total Costs", inplace = True, axis = 1)
        X = test.to_numpy()
        X_new = trans.fit_transform(X)
        X = X_new.T[reg.coef_ != 0].T
        dummy = np.array([1 for _ in range(X.shape[0])]).reshape(-1,1)
        X = np.concatenate((X,dummy), axis = 1)
        # import pdb; pdb.set_trace()

        print(lam, np.linalg.norm(np.matmul(X, W)- Y)/ np.linalg.norm(Y), X.shape[1])

    # import pdb; pdb.set_trace()

    

if __name__ == "__main__":
    args = sys.argv
    main(args)