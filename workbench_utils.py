import torch
import torch.nn as nn
import numpy as np
import random
import os
import signal

# 固定随机数种子
# set the random seed
def prepareEnv(seed = 1):
    import torch.backends.cudnn as cudnn

    # controls whether cuDNN is enabled. cudnn could accelerate the training procedure
    # cudnn.enabled = False
    cudnn.enabled = True
    # 使得每次返回的卷积算法是一样的
    # if True, causes cuDNN to only use deterministic convolution algorithms
    cudnn.deterministic = True
    # 如果网络的输入数据维度或类型上变化不大, 可以增加运行效率
    # 自动寻找最适合当前的高效算法,优化运行效率
    # if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest
    # cudnn.benchmark = False
    cudnn.benchmark = True
    

    """
    在需要生成随机数据的实验中，每次实验都需要生成数据。
    设置随机种子是为了确保每次生成固定的随机数，这就使得每次实验结果显示一致了，有利于实验的比较和改进。
    """
    torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数
    torch.cuda.manual_seed_all(seed)  # 给所有GPU设置
    np.random.seed(seed)
    random.seed(seed)
   


def get_rootdir_path():
    # 当前文件的路径
    current_path = os.path.abspath(__file__)

    # 当前文件所在的目录
    root_dir = os.path.dirname(current_path)
    return root_dir


def normalize(x, dataset_name, norm_type="n1"):
    if norm_type == "n1":
        # transfer learning
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif norm_type == "n2":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        raise RuntimeError(f"norm_type:{norm_type} is invalid")
    import torchvision.transforms as transforms
    trans_norm = transforms.Compose([
        transforms.Normalize(
            mean,
            std
        )
    ])
    return trans_norm(x)



def denormalize(x, norm_type="n1"):
    """
    inverse operation for transforms.Normalize
    it's useful when we need to visualize image data that have been normalized
    Args:
        x (Tensor): Float tensor image of size (C, H, W) or (B, C, H, W) to be normalized.
        dataset_name (str): it is used to select the specific values of mean and std
        norm_type (str): "n1" default. 
            "n1" indicates that we use the mean and std in pre-trained vggnet
            "n2" indicates that we use 0.5 as the mean and std
    """
    # selecting spcific mean and std for different dataset
    if norm_type == "n1":
        # transfer learning
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif norm_type == "n2":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        raise RuntimeError(f"norm_type:{norm_type} is invalid")

    mean = torch.as_tensor(mean, dtype=x.dtype, device=x.device)
    std = torch.as_tensor(std, dtype=x.dtype, device=x.device)
    if mean.ndim == 1:
        mean = mean.view((-1, 1, 1))
    if std.ndim == 1:
        std = std.view((-1, 1, 1))
    # the operation in F.normalize is: tensor.sub_(mean).div_(std)
    # so we can mutiply std and add mean
    # According to the broadcast mechanism, the shape of mean and std will be aligned with x
    x.mul_(std).add_(mean)
    return x
