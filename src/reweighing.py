from collections import Counter
import numpy as np

def Reweighing(features):
    N = len(features["Label"])
    weight = np.array([1.0]*N)
    PAY1 = 0
    PAY0 = 0
    AY1 = []
    AY0 = []
    for i in range(N):
        if np.all(features["Aij"][i]==[0,1]):
            if features["Label"][i] > 0:
                PAY1 += 1
                AY1.append(i)
            else:
                PAY0 += 1
                AY0.append(i)
    PAij = PAY1 + PAY0
    if PAY1 == 0 or PAY0 == 0:
        wAY1 = 1.0
        wAY0 = 0.0
    else:
        wAY1 = PAij / (2 * PAY1)
        wAY0 = PAij / (2 * PAY0)
    for index in AY1:
        weight[index] = wAY1
    for index in AY0:
        weight[index] = wAY0

    return weight