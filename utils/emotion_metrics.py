# THIS FILE IS BORROWED FROM https://github.com/face-analysis/emonet/

# NUMPY FUNCTIONS
import numpy as np


def ACC(ground_truth, predictions):
    """Evaluates the mean accuracy
    """
    return np.mean(ground_truth.astype(int) == predictions.astype(int))

def RMSE(ground_truth, predictions):
    """
        Evaluates the RMSE between estimate and ground truth.
    """
    return np.sqrt(np.mean((ground_truth-predictions)**2))


def SAGR(ground_truth, predictions):
    """
        Evaluates the SAGR between estimate and ground truth.
    """
    return np.mean(np.sign(ground_truth) == np.sign(predictions))


def PCC(ground_truth, predictions):
    """
        Evaluates the Pearson Correlation Coefficient.
        Inputs are numpy arrays.
        Corr = Cov(GT, Est)/(std(GT)std(Est))
    """
    return np.corrcoef(ground_truth, predictions)[0,1]


def CCC(ground_truth, predictions):
    """
        Evaluates the Concordance Correlation Coefficient.
        Inputs are numpy arrays.
    """
    mean_pred = np.mean(predictions)
    mean_gt = np.mean(ground_truth)

    std_pred= np.std(predictions)
    std_gt = np.std(ground_truth)

    pearson = PCC(ground_truth, predictions)
    return 2.0*pearson*std_pred*std_gt/(std_pred**2+std_gt**2+(mean_pred-mean_gt)**2)

def ICC(labels, predictions):
    """Evaluates the ICC(3, 1)
    """
    naus = predictions.shape[1]
    icc = np.zeros(naus)

    n = predictions.shape[0]

    for i in range(0,naus):
        a = np.asmatrix(labels[:,i]).transpose()
        b = np.asmatrix(predictions[:,i]).transpose()
        dat = np.hstack((a, b))
        mpt = np.mean(dat, axis=1)
        mpr = np.mean(dat, axis=0)
        tm  = np.mean(mpt, axis=0)
        BSS = np.sum(np.square(mpt-tm))*2
        BMS = BSS/(n-1)
        RSS = np.sum(np.square(mpr-tm))*n
        tmp = np.square(dat - np.hstack((mpt,mpt)))
        WSS = np.sum(np.sum(tmp, axis=1))
        ESS = WSS - RSS
        EMS = ESS/(n-1)
        icc[i] = (BMS - EMS)/(BMS + EMS)

    return icc

# TORCH FUNCTIONS
import torch


def ACC_torch(ground_truth, predictions):
    """Evaluates the mean accuracy
    """
    assert ground_truth.shape == predictions.shape
    return torch.mean( torch.eq(ground_truth.int(), predictions.int()).float())

def RMSE_torch(ground_truth, predictions):
    """
        Evaluates the RMSE between estimate and ground truth.
    """
    assert ground_truth.shape == predictions.shape
    return torch.sqrt(torch.mean(torch.pow((ground_truth-predictions), 2)))


def SAGR_torch(ground_truth, predictions):
    """
        Evaluates the SAGR between estimate and ground truth.
    """
    assert ground_truth.shape == predictions.shape
    return torch.mean( torch.eq(torch.sign(ground_truth), torch.sign(predictions)).float())


def PCC_torch(ground_truth, predictions, batch_first=True):
    """
        Evaluates the Pearson Correlation Coefficient.
        Inputs are numpy arrays.
        Corr = Cov(GT, Est)/(std(GT)std(Est))
    """
    # print(ground_truth.shape)
    # print(predictions.shape)
    assert ground_truth.shape == predictions.shape
    assert predictions.numel() >= 2 # std doesn't make sense, unless there is at least two items in the batch

    if batch_first:
        dim = -1
    else:
        dim = 0

    centered_x = ground_truth - ground_truth.mean(dim=dim, keepdim=True)
    centered_y = predictions - predictions.mean(dim=dim, keepdim=True)

    covariance = (centered_x * centered_y).sum(dim=dim, keepdim=True)

    bessel_corrected_covariance = covariance / (ground_truth.shape[dim] - 1)

    x_std = ground_truth.std(dim=dim, keepdim=True)
    y_std = predictions.std(dim=dim, keepdim=True)

    corr = bessel_corrected_covariance / (x_std * y_std)
    return corr


def CCC_torch(ground_truth, predictions, batch_first=False):
    """
        Evaluates the Concordance Correlation Coefficient.
        Inputs are numpy arrays.
    """
    assert ground_truth.shape == predictions.shape
    assert predictions.numel() >= 2  # std doesn't make sense, unless there is at least two items in the batch

    mean_pred = torch.mean(predictions)
    mean_gt = torch.mean(ground_truth)

    std_pred = torch.std(predictions)
    std_gt = torch.std(ground_truth)

    pearson = PCC_torch(ground_truth, predictions, batch_first=batch_first)
    return 2.0*pearson*std_pred*std_gt/(std_pred**2+std_gt**2+(mean_pred-mean_gt)**2)


def ICC_torch(labels, predictions):
    """Evaluates the ICC(3, 1)
    """
    assert ground_truth.shape == predictions.shape
    naus = predictions.shape[1]
    icc = torch.zeros(naus)

    n = predictions.shape[0]

    for i in range(0,naus):
        # a = np.asmatrix(labels[:,i]).transpose()
        a = labels[:,i:i+1]#.transpose(0,1)
        # b = np.asmatrix(predictions[:,i]).transpose()
        b = predictions[:,i:i+1]#.transpose(0,1)
        dat = torch.hstack((a, b))
        mpt = torch.mean(dat, dim=1, keepdim=True)
        mpr = torch.mean(dat, dim=0, keepdim=True)
        tm  = torch.mean(mpt, dim=0, keepdim=True)
        BSS = torch.sum(torch.square(mpt-tm))*2
        BMS = BSS/(n-1)
        RSS = torch.sum(torch.square(mpr-tm))*n
        tmp = torch.square(dat - torch.hstack((mpt, mpt)))
        WSS = torch.sum(torch.sum(tmp, dim=1))
        ESS = WSS - RSS
        EMS = ESS/(n-1)
        icc[i] = (BMS - EMS)/(BMS + EMS)

    return icc
