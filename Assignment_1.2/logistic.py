import sys
import pandas as pd
import numpy as np
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax

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

    return X_train, Y_train, X_test

def load_data_for_d(train_file_name, test_file_name):
    train = pd.read_csv(train_file_name, index_col = 0)    
    test = pd.read_csv(test_file_name, index_col = 0)
    col_drop = ["Payment Typology 1","Gender","APR Medical Surgical Description","Payment Typology 3","Ethnicity","Race","Zip Code - 3 digits","Hospital County","Facility Id","Operating Certificate Number","Facility Name","Health Service Area","Type of Admission"]   
    train.drop(columns = col_drop, inplace = True)
    test.drop(columns = col_drop, inplace = True)

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

    return X_train, Y_train, X_test

def get_param_dict_a(param_file):

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

def get_param_dict_b(param_file):

    return_dict = {"mode": None,
                    "n0": None,
                    "alpha":None,
                    "beta": None,
                    "n_iter": None,
                    "bs": None}

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
        bs = int(lines[3])
        return_dict["n_iter"] = n_iter
        return_dict["bs"] = bs

    print(return_dict)
    return return_dict

def get_lr(param_dict, t, W, X_train, Y_train, d, n0):
    # t is the iteration number of gradient update
    mode = param_dict["mode"]
    if(mode == 1):
        return param_dict["n0"]
    elif(mode == 2):
        return param_dict["n0"]/(t**0.5)
    elif(mode == 3):
        # dunno how to do this
        alpha = param_dict["alpha"]
        beta = param_dict["beta"]

        lr = n0
        # import pdb; pdb.set_trace()
        baseline_loss = loss_given_weight(X_train, Y_train, W)
        while(loss_given_weight(X_train, Y_train, W - lr*d) > baseline_loss - alpha*lr*np.linalg.norm(d)**2 ):
            lr*=beta

        return lr

def loss(Y_train, Y_hat_train):
    return -(np.log(Y_hat_train)*Y_train).sum()/(Y_train.shape[0])

def loss_given_weight(X_train, Y_train, W):
    gamma = 10**(-15)
    return -(np.log(np.clip(softmax(np.matmul(X_train, W), axis = 1), gamma, 1-gamma))*Y_train).sum()/(Y_train.shape[0])

        
def get_lr_for_part_c(t, W, X_train, Y_train, d, n0):
    return 0.25
        


def mode_a(args):
    train_file_name, test_file_name, param_file, output_file_name, weight_file_name = args[2:] 
    X_train, Y_train, X_test = load_data(train_file_name, test_file_name)
    # Y_train shape is [batch,k]
    assert X_train.shape[1] == X_test.shape[1]

    param_dict = get_param_dict_a(param_file)
    n_iter = param_dict["n_iter"]
    lr = param_dict["n0"]
    n0 = param_dict["n0"]

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
        lr = get_lr(param_dict,t, W, X_train, Y_train, d, n0)
        # print(Y_hat_train.argmax(axis = 1))

        W = W - lr*d

        # pbar.set_postfix({"Loss": loss(Y_train, Y_hat_train), "LR": lr})
    
    Y_hat_test = softmax(np.matmul(X_test, W), axis = 1).argmax(axis = 1)
    # 0-7 encoding

    Y_hat_test+=1
    # 1-8 encoding

    with open(output_file_name, "w") as f:
        for y in Y_hat_test:
            f.write(str(y))
            f.write("\n")

    with open(weight_file_name, "w") as f:
        for w in W.reshape(-1):
            f.write(str(w))
            f.write("\n")

    # import pdb; pdb.set_trace()
    pass

def mode_b(args):
    train_file_name, test_file_name, param_file, output_file_name, weight_file_name = args[2:] 
    X_train, Y_train, X_test = load_data(train_file_name, test_file_name)
    # Y_train shape is [batch,k]
    assert X_train.shape[1] == X_test.shape[1]

    param_dict = get_param_dict_b(param_file)
    n_iter = param_dict["n_iter"]
    lr = param_dict["n0"]
    n0 = param_dict["n0"]
    bs = param_dict["bs"]

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
        lr = get_lr(param_dict,t, W, X_train, Y_train, d, n0)
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

    
    Y_hat_test = softmax(np.matmul(X_test, W), axis = 1).argmax(axis = 1)
    # 0-7 encoding

    Y_hat_test+=1
    # 1-8 encoding

    with open(output_file_name, "w") as f:
        for y in Y_hat_test:
            f.write(str(y))
            f.write("\n")

    with open(weight_file_name, "w") as f:
        for w in W.reshape(-1):
            f.write(str(w))
            f.write("\n")

    # import pdb; pdb.set_trace()
    pass

def mode_c(args):
    train_file_name, test_file_name, output_file_name, weight_file_name = args[2:] 
    X_train, Y_train, X_test = load_data(train_file_name, test_file_name)
    # Y_train shape is [batch,k]
    assert X_train.shape[1] == X_test.shape[1]

    # param_dict = get_param_dict_b(param_file)
    n_iter = 500
    lr = 0.25
    n0 = 0.25
    bs = 1000


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
        lr = get_lr_for_part_c(t, W, X_train, Y_train, d, n0)
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

        if(t%50 == 0):
            Y_hat_test = softmax(np.matmul(X_test, W), axis = 1).argmax(axis = 1)
            # 0-7 encoding

            Y_hat_test+=1
            # 1-8 encoding

            with open(output_file_name, "w") as f:
                for y in Y_hat_test:
                    f.write(str(y))
                    f.write("\n")

            with open(weight_file_name, "w") as f:
                for w in W.reshape(-1):
                    f.write(str(w))
                    f.write("\n")
    

    # import pdb; pdb.set_trace()
    pass

def mode_d(args):
    train_file_name, test_file_name, output_file_name, weight_file_name = args[2:] 
    X_train, Y_train, X_test = load_data_for_d(train_file_name, test_file_name)
    # index = [1, 4, 5, 6, 10, 62, 78, 88, 92, 108, 109, 111, 147, 148, 158, 184, 187, 188, 192, 193, 199, 200, 206, 208, 211, 215, 216, 218, 219, 220, 221, 234, 253, 265, 266, 288, 289, 292, 333, 343, 344, 369, 370, 375, 380, 381, 382, 385, 388, 389, 390, 397, 400, 402, 403, 406, 407, 408, 410, 412, 413, 414, 431, 439, 440, 442, 443, 444, 451, 453, 456, 462, 473, 474, 478, 486, 488, 498, 501, 503, 504, 506, 513, 514, 516, 520, 522, 528, 533, 536, 542, 559, 560, 565, 594, 613, 615, 623, 626, 627, 630, 644, 647, 651, 655, 656, 657, 658, 669, 708, 710, 713, 714, 715, 716, 723, 726, 727, 729, 730, 731, 732, 734, 736, 737, 738, 748, 749, 750, 752, 753, 754, 759, 764, 771, 773, 774, 775, 777, 780, 781, 790, 793, 807, 808, 817, 819, 820, 822, 823, 825, 826, 830, 832, 833, 834, 836, 837, 839, 846, 848, 849, 850, 851, 852, 853, 854, 855, 860, 862, 866, 870, 876, 880, 882, 883, 884, 889, 893, 899, 901, 902, 903, 904, 905, 907, 908, 909, 910, 911, 913, 914, 915, 917, 919, 921, 923, 935, 939, 943, 946, 948, 950, 952, 953, 954, 955, 961, 962, 965, 968, 973, 974, 979, 980, 984, 986, 987, 988, 993, 995, 999, 1002, 1021, 1025, 1028, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1039, 1041, 1044, 1045, 1047, 1050, 1053, 1054, 1058, 1059, 1060, 1061, 1062, 1066, 1067, 1068, 1078, 1080, 1081, 1084, 1086, 1087, 1095, 1100, 1101, 1102, 1110, 1111, 1115, 1117, 1118, 1119, 1120, 1121, 1123, 1125, 1128, 1133, 1134, 1135, 1138, 1139, 1140, 1141, 1148, 1151, 1152, 1155, 1156, 1190, 1191, 1193, 1194, 1196, 1197, 1198, 1200, 1201, 1202, 1206, 1208, 1211, 1212, 1213, 1215, 1216, 1218, 1219, 1223, 1225, 1226, 1228, 1230, 1234, 1235, 1238, 1239, 1249, 1252, 1253, 1255, 1257, 1258, 1261, 1265, 1266, 1267, 1268, 1269, 1272, 1274, 1275, 1276, 1277, 1278, 1279, 1283, 1285, 1290, 1291, 1293, 1296, 1297, 1299, 1300, 1301, 1305, 1306, 1308, 1310, 1314, 1315, 1319, 1323, 1324, 1325, 1328, 1329, 1330, 1331, 1332, 1333, 1340, 1341, 1342, 1343, 1344, 1345, 1350, 1351, 1355, 1357, 1359, 1360, 1366, 1372, 1373, 1376, 1377, 1389, 1392, 1393, 1396, 1397, 1398, 1400, 1402, 1404, 1410, 1411, 1414, 1418, 1419, 1420, 1423, 1424, 1425, 1427, 1429, 1434, 1435, 1437, 1438, 1440, 1441, 1443, 1445, 1447, 1448, 1452, 1453, 1455, 1461, 1464, 1465, 1467, 1469, 1470, 1471, 1477, 1478, 1480, 1481, 1484, 1485, 1488, 1489, 1490, 1491, 1492, 1495, 1496, 1505, 1507, 1511, 1513, 1514, 1515, 1517, 1518, 1519, 1520, 1521, 1522, 1523, 1524, 1525, 1526, 1527, 1528, 1529, 1531, 1532, 1533, 1534, 1535, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1549, 1550, 1551, 1552, 1553, 1558, 1559, 1561, 1562, 1563, 1568, 1569, 1571, 1572, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1632]
    index = [1, 2, 3, 4, 5, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 33, 34, 35, 36, 38, 39, 40, 45, 50, 53, 56, 57, 59, 60, 61, 63, 66, 67, 71, 76, 77, 79, 80, 87, 88, 93, 94, 99, 103, 105, 106, 108, 109, 111, 112, 116, 118, 119, 120, 122, 123, 125, 132, 134, 135, 136, 137, 138, 139, 140, 141, 146, 148, 152, 155, 156, 159, 160, 162, 163, 164, 166, 168, 169, 170, 173, 175, 176, 179, 181, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 199, 200, 201, 203, 205, 207, 208, 209, 210, 221, 225, 229, 230, 232, 233, 234, 236, 237, 238, 239, 240, 241, 243, 245, 247, 248, 251, 252, 254, 255, 256, 259, 260, 261, 262, 265, 266, 270, 272, 273, 274, 279, 281, 282, 285, 287, 288, 304, 307, 310, 311, 314, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 330, 331, 333, 334, 336, 337, 338, 339, 340, 344, 345, 346, 347, 348, 349, 350, 352, 353, 354, 357, 358, 359, 362, 364, 366, 367, 368, 370, 371, 372, 373, 375, 376, 379, 381, 383, 384, 385, 386, 387, 388, 391, 396, 397, 401, 402, 403, 404, 405, 406, 407, 409, 411, 412, 414, 419, 420, 421, 424, 425, 426, 427, 434, 436, 437, 438, 440, 441, 442, 443, 459, 465, 475, 476, 477, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 492, 494, 496, 497, 498, 499, 501, 502, 504, 505, 507, 508, 509, 511, 512, 514, 516, 520, 521, 524, 525, 533, 535, 536, 538, 539, 541, 542, 543, 544, 545, 547, 550, 551, 552, 553, 554, 555, 558, 560, 561, 562, 563, 564, 565, 566, 569, 570, 571, 574, 575, 576, 577, 579, 582, 583, 585, 586, 587, 591, 592, 594, 595, 596, 599, 600, 601, 602, 604, 605, 608, 609, 610, 611, 614, 615, 616, 617, 618, 619, 620, 622, 623, 626, 627, 628, 629, 630, 631, 636, 637, 641, 643, 645, 646, 648, 650, 651, 652, 653, 655, 658, 659, 662, 663, 665, 667, 671, 675, 677, 678, 679, 682, 683, 684, 686, 688, 690, 696, 697, 700, 703, 704, 705, 706, 707, 709, 710, 711, 712, 713, 715, 717, 719, 720, 721, 723, 724, 726, 727, 729, 731, 733, 734, 735, 737, 738, 739, 741, 747, 749, 750, 751, 753, 755, 756, 757, 758, 760, 763, 764, 766, 767, 770, 771, 774, 775, 776, 777, 778, 781, 782, 785, 787, 791, 792, 793, 797, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 832, 834, 835, 837, 838, 839, 848, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 898]
    X_train = X_train[: ,index]
    X_test = X_test[: ,index]
    # Y_train shape is [batch,k]
    assert X_train.shape[1] == X_test.shape[1]

    # param_dict = get_param_dict_b(param_file)
    n_iter = 500
    lr = 0.25
    n0 = 0.25
    bs = 1000


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
        lr = get_lr_for_part_c(t, W, X_train, Y_train, d, n0)
        for batch in range((X_train.shape[0]-1)//bs + 1):
            # print(batch)
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

        if(t%50 == 0):
            Y_hat_test = softmax(np.matmul(X_test, W), axis = 1).argmax(axis = 1)
            # 0-7 encoding

            Y_hat_test+=1
            # 1-8 encoding

            with open(output_file_name, "w") as f:
                for y in Y_hat_test:
                    f.write(str(y))
                    f.write("\n")

            with open(weight_file_name, "w") as f:
                for w in W.reshape(-1):
                    f.write(str(w))
                    f.write("\n")
    

    # import pdb; pdb.set_trace()
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