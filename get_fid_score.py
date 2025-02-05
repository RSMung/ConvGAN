import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


import numpy as np
from scipy.linalg import sqrtm

# from inception import InceptionV3
from torchvision.models import inception_v3, Inception_V3_Weights

"""
https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
"""

@torch.no_grad()
def get_fid_score(realData, generatedData, device):
    """
    计算fid score
    :param realData: 真实图片数据   [B, 1, h, w]
    :param generatedData: 生成的图片的数据   [B, 1, h, w]
    :return: fid_score
    """
    # 1. 加载预训练的Inception v3模型
    # model = inception_v3(pretrained=True, transform_input=False, aux_logits=False).to(device)
    model = inception_v3(
        weights=Inception_V3_Weights.IMAGENET1K_V1, 
        transform_input=False
    ).to(device)
    # 因为只用来提取特征, 因此去除最后的dropout和fc
    model.dropout = nn.Identity()
    model.fc = nn.Identity()

    # 计算均值和协方差矩阵
    mu1, sigma1 = get_statistics(realData, model, device)
    mu2, sigma2 = get_statistics(generatedData, model, device)

    fid_score = computeFidScore(mu1, sigma1, mu2, sigma2)
    return fid_score

@torch.no_grad()
def get_statistics(data, model, device):
    features = get_features(data, model, device)
    # mu = np.mean(features, axis=0)
    # sigma = np.cov(features, rowvar=False)
    mu = torch.mean(features, dim=0)
    # pytorch 与 numpy的协方差矩阵计算方式不同
    # https://blog.csdn.net/Yonggie/article/details/124757929
    sigma = torch.cov(features.T)
    return mu, sigma

@torch.no_grad()
def computeFidScore(mu1, sigma1, mu2, sigma2):
    mu1 = np.atleast_1d(mu1.cpu().numpy())
    mu2 = np.atleast_1d(mu2.cpu().numpy())

    sigma1 = np.atleast_2d(sigma1.cpu().numpy())
    sigma2 = np.atleast_2d(sigma2.cpu().numpy())
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        # 协方差矩阵出现了无穷大
        # np.eye生成对角阵
        offset = np.eye(sigma1.shape[0]) * 1e-6
        # 添加一个偏置再计算协方差矩阵
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # 可能出现复数
    if np.iscomplexobj(covmean):
        # np.diagonal获取矩阵对角线上的元素
        # np.allclose比较两个array是不是每一元素都相等
        # https://www.shujudaka.com/documents/data-analysis-basic/numpy-function/numpy-allclose.html
        # if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
        #     # covmean矩阵的对角线上不全为0
        #     m = np.max(np.abs(covmean.imag))
        #     raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

@torch.no_grad()
def get_features(data, model, device):
    model.eval()
    dataLoader = DataLoader(data,batch_size=8)
    # pred_arr = np.empty((data.shape[0], 2048))
    pred_arr = None
    # start_idx = 0
    for batch_img in iter(dataLoader):
        batch_img = batch_img.to(device)  # [b, c, h, w]
        if batch_img.shape[1] == 1:
            batch_img = batch_img.repeat(1, 3, 1, 1)  # [b, 3, h, w]
        if batch_img.shape[2] < 84:
            # (128, 128)
            # (96, 96)
            # (84, 84)
            # the original input size of inception v3 is 299
            batch_img = F.interpolate(batch_img, (299, 299))
        # print(batch_img.shape)
        pred = model(batch_img)
        # print(pred.shape)  # [B, 2048]
        if pred_arr is None:
            pred_arr = pred
        else:
            pred_arr = torch.cat((pred_arr, pred), dim=0)
        # # pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        # start_idx = start_idx + pred.shape[0]
    return pred_arr

if __name__ == "__main__":
    import time
    # 随机数矩阵计算fid示例
    # b = 2
    # b = 8
    b = 64
    # b = 10000
    # h = 224
    # w = 224
    h = 64
    w = 64
    realData = torch.randn((b, 3, h, w))
    generatedData = torch.rand((b, 3, h, w))
    mydevice = "cuda"

    start_time = time.time()
    fid_score = get_fid_score(realData, generatedData, mydevice)
    end_time = time.time()
    print('fid_score', fid_score)
    print('cost time', end_time - start_time)


    # fid_score = get_fid_score(realData, realData, mydevice)
    # print(fid_score)