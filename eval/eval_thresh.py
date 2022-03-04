import pickle
import numpy as np


if __name__ == '__main__':
    with open('./distance_00_mmdf.pickle', 'rb') as f:
        data = pickle.load(f)
    distance = data[0]
    sim = data[1]
    print(distance)
    print(sim)
    print(len(distance))
    print(len(sim))
    # thresh = range(1, 31)
    # thresh = np.linspace(1, 40, num=50)
    thresh = np.linspace(0.2, 3, num=50)
    # thresh = np.linspace(0.015, 0.04, num=50)
    recall = []
    precision = []
    for ths in thresh:
        p = 0
        n = 0
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for i in range(len(distance)):
            # print(distance[i])
            predict = distance[i] < ths
            # print(predict)
            # print(sim[i], bool(sim[i]))
            if bool(sim[i]):
                p = p + 1
                if predict == bool(sim[i]):
                    tp = tp + 1
                else:
                    fn = fn + 1
            else:
                n = n + 1
                if predict == bool(sim[i]):
                    tn = tn + 1
                else:
                    fp = fp + 1
                    # print('fp:%d' % i)
        if p is not 0 and (tp + fp) is not 0:
            print('%f, %f, %f' % (ths, tp/p, tp/(tp+fp)))
            recall.append(tp/p)
            precision.append(tp/(tp+fp))
    # with open('../result/pr_00_mmdf.pickle', 'wb') as f:
    #     pickle.dump([recall, precision], f)
