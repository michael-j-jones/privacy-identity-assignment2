import numpy as np
import pandas as pd

# Measurements are labeled as A, B, C, D
measurementID = ['A', 'B', 'C', 'D']
pd.options.mode.chained_assignment = None

def lowest_percentage_match(traindata, testdata):
    percentagedf = pd.DataFrame()
    percentagedf['TrueID'] = testdata['ID']
    percentagedf['PercentSumOff'] = 100000
    percentagedf['PredictedID'] = ''
    for k in range(len(testdata['ID'])):
        saved_pct = 1000
        saved_ID = testdata.ID[k]
        for i in range(len(traindata['ID'])):
            a_pct = np.abs(testdata.A[k] - traindata.A[i]) / testdata.A[k]
            b_pct = np.abs(testdata.B[k] - traindata.B[i]) / testdata.B[k]
            c_pct = np.abs(testdata.C[k] - traindata.C[i]) / testdata.C[k]
            d_pct = np.abs(testdata.D[k] - traindata.D[i]) / testdata.D[k]
            sum_pct = a_pct + b_pct + c_pct + d_pct

            if sum_pct < saved_pct:
                saved_ID = traindata['ID'][i]
                saved_pct = sum_pct

        percentagedf.PercentSumOff[k] = saved_pct
        percentagedf.PredictedID[k] = saved_ID

    return percentagedf



if __name__ == '__main__':
    # Training data, sample 1
    traindf = pd.read_csv('train.csv')
    # Testing data, sample 2
    testdf = pd.read_csv('test.csv')


    percentagedf = lowest_percentage_match(traindf, testdf)
