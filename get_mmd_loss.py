import torch

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params: 
	    source: 源域数据[n, len(x)]
	    target: 目标域数据[m, len(y)]
	    kernel_mul: 
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 如果该参数不为None则使用该值作为高斯核的固定sigma值
	Return:
		sum(kernel_val): 多个核矩阵之和
    '''
    # 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    n_samples = int(source.size()[0])+int(target.size()[0])
    #将source,target按列方向合并
    total = torch.cat([source, target], dim=0)   # [n+m, feats_dim]
    """
    将total复制 n+m 份
    假设矩阵total是三个行向量a b c
    这里做的复制是整体复制:
    即从[a, b, c]变成
    [
        [a, b, c], 
        [a, b, c], 
        [a, b, c]
    ]
    """
    total0 = total.unsqueeze(0)   # [1, n+m, feats_dim]
    total0 = total0.expand(
        int(total.size(0)),
        int(total.size(0)), 
        int(total.size(1))
    )   # [n+m, n+m, feats_dim]
    """
    将total的每一行都复制成 n+m 行, 即每个数据都扩展成 n+m 份
    假设矩阵total是三个行向量a b c
    这里做的复制是每个行向量复制n+m份:
    即从[a, b, c]变成
    [
        [a, a, a], 
        [b, b, b], 
        [c, c, c]
    ]
    """
    total1 = total.unsqueeze(1)   # [n+m, 1, feats_dim]
    total1 = total1.expand(
        int(total.size(0)),    # n+m
        int(total.size(0)),    # n+m
        int(total.size(1))    # feats_dim
    )   # [n+m, n+m, feats_dim]
    """
    求任意两个数据之间的和 得到的矩阵中坐标 (i,j) 代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0)
    [                           [
        [a, b, c],                  [a, a, a], 
        [a, b, c],                  [b, b, b], 
        [a, b, c]                   [c, c, c]
    ]                           ]
    """
    L2_distance = ((total0-total1)**2).sum(2)   # [n+m, n+m, feats_dim] -> [n+m, n+m]
    #调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        # 计算一个初始值
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    # 以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值
    # 比如fix_sigma为1, kernel_mul=2.0时，得到[0.25, 0.5, 1, 2, 4]
    bandwidth /= kernel_mul ** (kernel_num // 2)   # 0.25
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    # 高斯核函数的数学表达式 e^(-l2_distance/sigma)
    # list中每个元素都是 [n+m, n+m] 的矩阵
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    #得到最终的核矩阵
    return sum(kernel_val)#/len(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    rbf指的是Radial Basis Function Kernel (径向基函数核)
    K(x, y) = e^(-gamma * ||x-y||^2)
    Params: 
	    source: 源域数据[n, len(x)]
	    target: 目标域数据[m, len(y)]
	    kernel_mul: 
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		loss: MMD loss
    '''
    batch_size = int(source.size()[0])#一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(
        source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma
    )
    #根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss#因为一般都是n==m，所以L矩阵一般不加入计算
