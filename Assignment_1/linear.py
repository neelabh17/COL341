import sys
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder

class_num_dict_tmp = [['Health Service Area', 8], ['Hospital County', 57], ['Facility Name', 212], ['Age Group', 5], ['Zip Code - 3 digits', 50], ['Gender', 3], ['Race', 4], ['Ethnicity', 4], ['Type of Admission', 6], ['Patient Disposition', 19], ['CCS Diagnosis Description', 260], ['CCS Procedure Description', 224], ['APR DRG Description', 308], ['APR MDC Description', 24], ['APR Severity of Illness Description', 4], ['APR Risk of Mortality', 4], ['APR Medical Surgical Description', 2], ['Payment Typology 1', 10], ['Payment Typology 2', 11], ['Payment Typology 3', 11], ['Emergency Department Indicator', 2]]
def transforms(X, trans, enc, do_reg = False, df_x = None, df_y = None):
    # polynimial fitting
    X = trans.fit_transform(X)
    













    # one hot encoding
    list_to_encode = ["Patient Disposition", "Age Group", "Payment Typology 1", "Payment Typology 2", "Payment Typology 3"]
    # for cls_name in tqdm(enc):
    for cls_name in list_to_encode:
        # print("\nOne hoting for ", cls_name)
        one_hot = enc[cls_name].transform(df_x[cls_name].to_numpy().reshape(-1,1)).toarray()
        # import pdb; pdb.set_trace()
        X = np.concatenate((X, one_hot ), axis = 1)
    
    # print("After One hot")
    # print(X.shape)
    #TODO add ignore
    ignore = []
    X = X.T[ignore != 0].T
        


    feat = np.zeros((df_x.shape[0],1)) + np.random.rand(df_x.shape[0],1)/1000
    feat[(df_x["Birth Weight"] <=1000) & (df_x["Birth Weight"] !=0) ] = 1
    X = np.concatenate((X, feat ), axis = 1)

    feat = np.zeros((df_x.shape[0],1)) + np.random.rand(df_x.shape[0],1)/1000
    feat[(df_x["Birth Weight"] <=1500) & (df_x["Birth Weight"] >=1000) ] = 1 
    X = np.concatenate((X, feat ), axis = 1)

    feat = np.zeros((df_x.shape[0],1)) + np.random.rand(df_x.shape[0],1)/1000
    feat[(df_x["Birth Weight"] <=2000) & (df_x["Birth Weight"] >1500) ] = 1
    X = np.concatenate((X, feat ), axis = 1)

    feat = np.zeros((df_x.shape[0],1)) + np.random.rand(df_x.shape[0],1)/1000
    feat[(df_x["Birth Weight"] <=3500) & (df_x["Birth Weight"] >2000) ] = 1
    X = np.concatenate((X, feat ), axis = 1)

    feat = np.zeros((df_x.shape[0],1)) + np.random.rand(df_x.shape[0],1)/1000
    feat[(df_x["Birth Weight"] >=3500) ] = 1
    X = np.concatenate((X, feat ), axis = 1)

    # Length of Stay 
    feat = np.zeros((df_x.shape[0],1)) + np.random.rand(df_x.shape[0],1)/1000
    feat[(df_x["Length of Stay"] >= 120 ) ] = 1
    X = np.concatenate((X, feat ), axis = 1)

    feat = np.zeros((df_x.shape[0],1)) + np.random.rand(df_x.shape[0],1)/1000
    feat[(df_x["Length of Stay"] <=15) & (df_x["Length of Stay"] >=5) ] = 1
    X = np.concatenate((X, feat ), axis = 1)

    # dummy = np.array([1 for _ in range(X.shape[0])]).reshape(-1,1)
    # X = np.concatenate((X,dummy), axis = 1)
    return X
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
    train_file_name, test_file_name, output_file_name = args[2:] 

    
    train_data = pd.read_csv(train_file_name)
    # train_data.drop("Total Costs", inplace = True, axis = 1)
    train_data.drop("Unnamed: 0", inplace = True, axis = 1)
    train_data.drop("Facility Id", inplace = True, axis = 1)
    train_data.drop("CCS Diagnosis Code", inplace = True, axis = 1)
    train_data.drop("CCS Procedure Code", inplace = True, axis = 1)
    train_data.drop("APR DRG Code", inplace = True, axis = 1)
    train_data.drop("APR MDC Code", inplace = True, axis = 1)
    train_data.drop("APR Severity of Illness Code", inplace = True, axis = 1)

    # train_data["Birth Weight"] =(train_data["Birth Weight"] - train_data["Birth Weight"].mean()) / train_data["Birth Weight"].std()
    # train_data["Length of Stay"] =(train_data["Length of Stay"] - train_data["Length of Stay"].mean()) / train_data["Length of Stay"].std()
    mean = train_data["Total Costs"].mean()
    std = train_data["Total Costs"].std()
    # print("Mean and std are ", mean, std)
    # train_data["Total Costs"] =(train_data["Total Costs"] - train_data["Total Costs"].mean()) / train_data["Total Costs"].std()

    class_num_dict = {}
    for a,b in class_num_dict_tmp:
        class_num_dict[a] = b
    enc = {}
    for class_name in class_num_dict:
        enc[class_name] = OneHotEncoder(handle_unknown='ignore')
        enc[class_name].fit(train_data[class_name].to_numpy().reshape(-1, 1))
    
    train_val_y = train_data["Total Costs"]
    train_data.drop("Total Costs", inplace = True, axis = 1)
    train_val_x = train_data

    test_data = pd.read_csv(test_file_name)
    # test_data.drop("Total Costs", inplace = True, axis = 1)
    test_data.drop("Unnamed: 0", inplace = True, axis = 1)
    test_data.drop("Facility Id", inplace = True, axis = 1)
    test_data.drop("CCS Diagnosis Code", inplace = True, axis = 1)
    test_data.drop("CCS Procedure Code", inplace = True, axis = 1)
    test_data.drop("APR DRG Code", inplace = True, axis = 1)
    test_data.drop("APR MDC Code", inplace = True, axis = 1)
    test_data.drop("APR Severity of Illness Code", inplace = True, axis = 1)

    # test_data["Birth Weight"] =(test_data["Birth Weight"] - test_data["Birth Weight"].mean()) / test_data["Birth Weight"].std()
    # test_data["Length of Stay"] =(test_data["Length of Stay"] - test_data["Length of Stay"].mean()) / test_data["Length of Stay"].std()
    # print("Mean and std are ", mean, std)
    # test_data["Total Costs"] =(test_data["Total Costs"] - test_data["Total Costs"].mean()) / test_data["Total Costs"].std()

    class_num_dict = {}
    for a,b in class_num_dict_tmp:
        class_num_dict[a] = b
    enc = {}
    for class_name in class_num_dict:
        enc[class_name] = OneHotEncoder(handle_unknown='ignore')
        enc[class_name].fit(test_data[class_name].to_numpy().reshape(-1, 1))
    
    # test_y = test_data["Total Costs"]
    # test_data.drop("Total Costs", inplace = True, axis = 1)
    test_x = test_data


    Y = train_val_y.to_numpy()
    X = train_val_x.to_numpy()
    # print()
    # print(X.shape,Y.shape)


    # normal
    # W = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))

    # polynomial
    trans = PolynomialFeatures(degree=2)
    X = transforms(X, trans, enc, do_reg= True, df_x=train_val_x)
    # X_new = trans.fit_transform(X)
    # X = X_new
    # reg.fit(X_new, Y)
    # X = X_new.T[reg.coef_ != 0].T
    # # import pdb; pdb.set_trace()


    # dummy = np.array([1 for _ in range(X.shape[0])]).reshape(-1,1)
    # X = np.concatenate((X,dummy), axis = 1)
    W = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))

    # print("W shape ", W.shape)


    # import pdb; pdb.set_trace()


    # Y = test_y.to_numpy()
    X = test_x.to_numpy()
    X = transforms(X, Y, trans, enc, do_reg= False, df_x = test_x)
    # X_new = trans.fit_transform(X)
    # X = X_new.T[reg.coef_ != 0].T
    # dummy = np.array([1 for _ in range(X.shape[0])]).reshape(-1,1)
    # X = np.concatenate((X,dummy), axis = 1)
    # # import pdb; pdb.set_trace()

    Y_hat = np.matmul(X, W)
    with open(output_file_name, "w") as f:
        for y in Y_hat:
            f.write(str(y))
            f.write("\n")



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