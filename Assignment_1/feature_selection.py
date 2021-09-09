import sys
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures



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
    # data_Y = data["Total Costs"]
    # data_X = data
    print(train.shape, val.shape, test.shape)


    # Super basic baseline
    Y = train_val["Total Costs"].to_numpy()
    train_val.drop("Total Costs", inplace = True, axis = 1)
    X = train_val


    # Lasso LARS
    reg = linear_model.LassoLars(alpha=0.01)
    reg.fit(X, Y)


    # normal
    # W = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))

    # polynomial
    trans = PolynomialFeatures(degree=2)
    X_new = trans.fit_transform(X)
    X = X_new
    reg.fit(X_new, Y)
    X = X_new.T[reg.coef_ != 0].T
    
    import pdb; pdb.set_trace()


    # dummy = [1 for _ in range(train_val.shape[0])]
    # X["dummy"] = dummy
    # X = X.to_numpy()

    W = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))


    import pdb; pdb.set_trace()


    Y = test["Total Costs"].to_numpy()
    test.drop("Total Costs", inplace = True, axis = 1)
    X = test

    dummy = [1 for _ in range(test.shape[0])]
    X["dummy"] = dummy
    X = X.to_numpy()

    trans = PolynomialFeatures(degree=2)
    X_new = trans.fit_transform(X)
    X = X_new

    print(np.linalg.norm(np.matmul(X, W)- Y)/ np.linalg.norm((Y)))


    

if __name__ == "__main__":
    args = sys.argv
    main(args)