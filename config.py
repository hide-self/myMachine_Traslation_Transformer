import torch

"""
模型参数配置
"""
d_model=512 #相当于词嵌入矩阵的d，用于降维
dropout=0.1 #随机失活10%



"""
词表的配置
"""
# 源语言(英语)的词表大小
src_vocab_size=32000
# 目标语言(中文)的词表大小
tgt_vocab_size=32000


"""
训练配置
"""
# 一个批次的大小
batch_size=32
# 训练的总轮次
epoch_num=3
# 学习率
lr=3e-4

"""
解码和生成设置
这些参数控制模型生成输出时的行为，影响生成的句子质量。
"""


"""
文件路径和模型配置
这些参数用于定义文件路径和是否加载预训练模型的设置。
"""
data_dir = './data'
train_data_path = './data/json/train.json'
dev_data_path = './data/json/dev.json'
test_data_path = './data/json/test.json'

model_path = './weights/transformer_model.pth'
test_model_path = './run/train/exp/weights/best_bleu_26.30.pth'


"""
设备配置
这些参数用于配置模型运行的硬件设备。
"""
# 指定使用的GPU设备的ID。
# 指定设备ID的列表。
gpu_id = '0'
device_id = [0]
# set device
if gpu_id != '':
    device = torch.device(f"cuda:{gpu_id}")
else:
    device = torch.device('cpu')

