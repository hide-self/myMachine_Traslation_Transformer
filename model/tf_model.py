import math
from torch.autograd import Variable
import config
import torch
from torch import nn
import torch.nn.functional as F
import copy

DEVICE=config.device

# 词嵌入层
class Embeddings(nn.Module):
    def __init__(self,d_model,vocab):
        """
        词嵌入层初始化
        (输入的是一个v*1的矩阵，则词嵌入层就是一个d*v矩阵，输出为d*1矩阵，此时已经被有效降维)
        :param d_model:相当于d最终输出的维度数
        :param vocab:相当于v为词表大小，即独热向量维度数
        """
        super(Embeddings,self).__init__()
        # Embedding层，将v*1的独热编码向量，映射成d*1的词向量
        self.lut=nn.Embedding(vocab,d_model)
        # 存储输出词向量的维度数d
        self.d_model=d_model

    # 前向传播
    def forwrad(self,x):
        # 乘以sqrt(d)是Transformer原文的做法
        # 目的：使得嵌入向量的量级与后续残差/位置编码在同一大小尺度，稳定训练、加快收敛
        return self.lut(x)*math.sqrt(self.d_model)

# 位置编码层(公式见原理部分)
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout,max_len=5000,device=DEVICE):
        super(PositionalEncoding,self).__init__()
        self.dropout=nn.Dropout(p=dropout)  # 随机失活

        # 初始化一个大小为 L*d 的全0矩阵(形状与词嵌入矩阵相同)
        pe=torch.zeros(max_len,d_model,device=device)
        # 生成一个位置下标的tensor矩阵(每一行都是一个位置下标)
        position = torch.arange(0., max_len, device=DEVICE).unsqueeze(1)
        # 这里幂运算太多，我们使用exp和log来转换实现公式中pos下面要除以的分母（由于是分母，要注意带负号）
        div_term = torch.exp(torch.arange(0., d_model, 2, device=DEVICE) * -(math.log(10000.0) / d_model))

        # 位置编码奇偶位计算
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,0::2]=torch.cos(position*div_term)

        # 在最前面开始加1个维度，变成 1*L*d大小的矩阵(代码中相当于1*max_len*d_model大小的矩阵)
        # (这样做方便后续与一个batch的句子所有词的embedding批量相加)
        pe=pe.unsqueeze(0)
        # 将pe矩阵以持久的buffer状态存下(不会作为要训练的参数)
        self.register_buffer('pe', pe)

    def forward(self,x):    # 输入的数据x就是词嵌入向量
        # 将一个batch的句子所有词的embedding与已构建好的positional embeding相加
        # (这里按照该批次数据的最大句子长度来取对应需要的那些positional embedding值)
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)  # 随机失活

# 注意力机制
def attention(query,key,value,mask=None,dropout=None):
    """
    注意力机制
    :param query:Q矩阵，形状L*d_k
    :param key: K矩阵，形状L*d_k
    :param value:V矩阵，形状L*d_k
    :param mask:掩码矩阵，哪些部分为0，到时候scores就填充负无穷
    :param dropout:随机失活函数
    :return:
    """
    d_k=query.size(-1)

    # Q与K^T做矩阵相乘，然后除以根号下d_k
    scores=torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
    # tensor.transpose(dim1.dim2)表示将dim1与dim2两个维度交换

    if mask is not None:    # mask不是None
        scores.masked_fill(mask==0,1e-9)    # mask为0的元素位置，在scores中填充为负无穷

    # softmax函数，获得注意力机制得分矩阵
    # softmax按照列维度来进行处理
    # 这样就表示，以当前行作为查询，同一行中所有列作为索引，同一行中所有列的值加起来为1
    p_attn=F.softmax(scores,dim=-1)

    # dropout随机失活函数
    if dropout is not None:
        p_attn=dropout(p_attn)

    # 输出：注意力机制矩阵、注意力机制得分矩阵
    return torch.matmul(p_attn,value),p_attn

# 深拷贝N个模块
def clones(module, N):
    """克隆模型块，克隆的模型块参数不共享"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# 多头注意力机制
class MutiHeadedAttention(nn.Module):
    def __init__(self,h,d_model,dropout=0.1):
        """
        初始化多头注意力机制类
        :param h: h表示头数
        :param d_model: 词嵌入矩阵L*d中的d
        :param dropout: 随机失活概率
        """
        # 保证可以整除
        assert d_model%h==0
        # 通过d_model、h获得d_k，这样获得的d_k便于计算
        self.d_k=d_model//h
        #head数量
        self.h=h
        #4个全连接函数，供应 WQ、WK、WV矩阵和最后h个多头注意力矩阵concat之后进行变换的矩阵W_o
        self.linears=clones(nn.Linear(d_model,d_model),4)
        self.attn=None
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,query,key,value,mask=None):
        if mask is not None:
            mask=mask.unsqueeze(1)
        # query的第一个维度值为batch size
        nbatches=query.size(0)
        # 将embedding层乘以WQ，WK，WV矩阵(均为全连接)
        # 并将结果拆成h块，然后将第二个和第三个维度值互换(具体过程见上述解析)
        # 最后query、key、value，就是含有h个元素的列表，他们就是h
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        #调用attention函数
        x,self.attn=attention(query,key, value, mask,self.dropout)
        # 将h个多头注意力矩阵concat起来（注意要先把h变回到第三维的位置）
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # 使用self.linears中构造的最后一个全连接函数来存放变换后的矩阵进行返回
        return self.linears[-1](x)

