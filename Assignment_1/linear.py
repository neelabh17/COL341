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

    import pdb; pdb.set_trace()




def mode_b(args):
    pass

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