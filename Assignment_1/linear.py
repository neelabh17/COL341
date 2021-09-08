import sys
import pandas as pd
import numpy as np
def mode_a(args):
    train_file_name, test_file_name, output_file_name, weight_file_name = args[2:] 
    #TODO will the whole files be paseed like we did in java
    # print(train_file_name, test_file_name, output_file_name, weight_file_name) 

    train_data = pd.read_csv(train_file_name)
    Y = train_data["Total Costs"].to_numpy()
    train_data.drop("Total Costs", inplace = True, axis = 1)
    train_data.drop("Unnamed: 0", inplace = True, axis = 1)
    X = train_data
    dummy = [1 for _ in range(train_data.shape[0])]
    X["dummy"] = dummy

    X = X.to_numpy()


    W = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))
    test_data = pd.read_csv(test_file_name)
    test_data.drop("Unnamed: 0", inplace = True, axis = 1)
    X_test = test_data
    dummy = [1 for _ in range(test_data.shape[0])]
    X_test["dummy"] = dummy
    X_test = X_test.to_numpy()
    Y_test = np.matmul(X_test, W)

    with open(output_file_name, "w") as f:
        for y in Y_test:
            f.write(str(y))
            f.write("\n")
    with open(weight_file_name, "w") as f:
        f.write(str(W[-1]))
        f.write("\n")
        for w in W[:-1]:
            f.write(str(w))
            f.write("\n")



    # import pdb; pdb.set_trace()







def mode_b(args):
    train_file_name, test_file_name, reg_file, output_file_name, weight_file_name, best_param_file = args[2:] 
    #TODO will the whole files be paseed like we did in java
    # print(train_file_name, test_file_name, output_file_name, weight_file_name) 
    lambdas = np.genfromtxt(reg_file, delimiter = ",")
    print(lambdas)

    train_data = pd.read_csv(train_file_name)
    Y = train_data["Total Costs"].to_numpy()
    train_data.drop("Total Costs", inplace = True, axis = 1)
    train_data.drop("Unnamed: 0", inplace = True, axis = 1)
    X = train_data
    dummy = [1 for _ in range(train_data.shape[0])]
    X["dummy"] = dummy

    X = X.to_numpy()


    # create K fold validations
    K = 10
    bins = np.linspace(0, train_data.shape[0], 11, dtype = np.int)
    starts = bins[:-1]
    ends = bins[1:]

    lambda_errors = []
    for lam in lambdas:
        error = 0.0
        tot = 0
        for start, end in zip(starts, ends):
            val_X = X[start:end, :]
            val_Y = Y[start:end]

            train_X = np.concatenate((X[0:start, :] , X[end:, :]), axis = 0)
            train_Y = np.concatenate((Y[0:start] , Y[end:]), axis = 0)
            # print(train_X.shape, val_X.shape, X.shape[0], X.shape[0]//10)
            W = np.matmul(np.linalg.inv(np.matmul(train_X.T, train_X)+lam*np.eye(train_X.shape[1])), np.matmul(train_X.T, train_Y))
            Y_hat = np.matmul(val_X, W)
            error += (np.linalg.norm(Y_hat-val_Y))/(np.linalg.norm(val_Y))
            tot += 1
        lambda_errors.append([lam, error/tot])

    lambda_errors.sort(key = lambda x: x[1])
    print(lambda_errors)

    best_lam = lambda_errors[0][0]
    W = np.matmul(np.linalg.inv(np.matmul(X.T, X)+best_lam*np.eye(X.shape[1])), np.matmul(X.T, Y))

    # import pdb; pdb.set_trace()
    test_data = pd.read_csv(test_file_name)
    test_data.drop("Unnamed: 0", inplace = True, axis = 1)
    X_test = test_data
    dummy = [1 for _ in range(test_data.shape[0])]
    X_test["dummy"] = dummy
    X_test = X_test.to_numpy()
    Y_test = np.matmul(X_test, W)

    with open(output_file_name, "w") as f:
        for y in Y_test:
            f.write(str(y))
            f.write("\n")
    with open(weight_file_name, "w") as f:
        f.write(str(W[-1]))
        f.write("\n")
        for w in W[:-1]:
            f.write(str(w))
            f.write("\n")
    
    with open(best_param_file, "w") as f:
        f.write(str(best_lam))



    # import pdb; pdb.set_trace()

def mode_c(args):
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

if __name__ == "__main__":
    args = sys.argv
    main(args)