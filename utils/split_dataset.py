import os
import numpy as np


if __name__ == '__main__':
    file = 'TestDataPairs06.txt'
    pair_1 = []
    pair_2 = []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            path1, path2, label = line.split(' ')
            if label == '1':
                pair_1.append([path1, path2])
            elif label == '0':
                pair_2.append([path1, path2])

    print(len(pair_1), len(pair_2))
    count = len(pair_1) // 5
    choice = np.random.choice(len(pair_1), count, replace=False)
    # print(choice)
    train_pair_1 = np.asarray(pair_1)[choice]
    train_pair_1 = train_pair_1.tolist()
    test_pair_1 = []

    # print(len(choice), len(pair_1))
    for num in range(len(pair_1)):
        if num not in choice:
            test_pair_1.append(pair_1[num])

    choice = np.random.choice(len(pair_1), count, replace=False)
    train_pair_2 = np.asarray(pair_2)[choice]
    train_pair_2 = train_pair_2.tolist()
    test_pair_2 = []
    for num in range(len(pair_2)):
        if num not in choice:
            test_pair_2.append(pair_2[num])
    print(len(train_pair_1), len(test_pair_1))
    print(len(train_pair_2), len(test_pair_2))
    with open('Train_06_1_4.txt', 'w') as f:
        for i in range(len(train_pair_1)):
            f.write(train_pair_1[i][0])
            f.write(' ')
            f.write(train_pair_1[i][1])
            f.write(' ')
            f.write('1')
            f.write('\n')
            f.write(train_pair_2[i][0])
            f.write(' ')
            f.write(train_pair_2[i][1])
            f.write(' ')
            f.write('0')
            f.write('\n')
    with open('Test_06_1_4.txt', 'w') as f:
        for i in range(len(test_pair_1)):
            f.write(test_pair_1[i][0])
            f.write(' ')
            f.write(test_pair_1[i][1])
            f.write(' ')
            f.write('1')
            f.write('\n')
            f.write(test_pair_2[i][0])
            f.write(' ')
            f.write(test_pair_2[i][1])
            f.write(' ')
            f.write('0')
            f.write('\n')
