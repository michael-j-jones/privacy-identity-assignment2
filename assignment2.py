import numpy as np
import pandas as pd

# Measurements are labeled as A, B, C, D
measurementID = ['A', 'B', 'C', 'D']
a_weight = 1
b_weight = 0.25
c_weight = 0.8
d_weight = 0.5
pd.options.mode.chained_assignment = None

def lowest_percentage_match(traindata, testdata):
    percentagedf = pd.DataFrame()
    percentagedf['TrueID'] = testdata['ID']
    percentagedf['PercentSumOff'] = 100000.
    percentagedf['PredictedID'] = ''
    for k in range(len(testdata['ID'])):
        saved_pct = 1000
        saved_ID = testdata.ID[k]
        for i in range(len(traindata['ID'])):
            a_pct = np.abs(testdata.A[k] - traindata.A[i]) / testdata.A[k]
            b_pct = np.abs(testdata.B[k] - traindata.B[i]) / testdata.B[k]
            c_pct = np.abs(testdata.C[k] - traindata.C[i]) / testdata.C[k]
            d_pct = np.abs(testdata.D[k] - traindata.D[i]) / testdata.D[k]
            sum_pct = a_pct*a_weight + b_pct*b_weight + c_pct*c_weight + d_pct*d_weight

            if sum_pct < saved_pct:
                saved_ID = traindata['ID'][i]
                saved_pct = sum_pct
        percentagedf.PercentSumOff[k] = saved_pct
        percentagedf.PredictedID[k] = saved_ID


    return percentagedf

def FARtest(trueID, predictedID):
    tp = sum(trueID == predictedID)
    tn = 0
    fp = len(trueID) - tp
    fn = len(trueID) - tn

    acc = (tp + tn) / (tp + fp + tn + fn)
    err = (fp + fn) / (tp + fp + tn + fn)

    print('tp = ', tp)
    print('Accuracy: ', acc, ', Error: ', err)


if __name__ == '__main__':
    # Training data, sample 1
    traindf = pd.read_csv('train.csv')
    # Testing data, sample 2
    testdf = pd.read_csv('test.csv')

    percentagedf = lowest_percentage_match(traindf, testdf)
    FARtest(percentagedf['TrueID'], percentagedf['PredictedID'])
