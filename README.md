# 基于Transformer的机器翻译
原理部分请移步：https://github.com/hide-self/myTransformer_learning



## 数据集文件说明



整体的数据集文件放在了`./data`下。

![77319649820](imgs/1773196572074.png)



**1.corpus.ch、corpus.en**

数据集原文说明：

corpus.ch、corpus.en两个文件就是翻译前后的完整原文，一句话一行，两个文件每一行一一对应

数据集下载方式：

可以通过这个网址下载类似的数据集：https://www.statmt.org/wmt18/translation-task.html

往下翻到一个表格，我们找到支持 **zh-en** 的数据集，把网址复制出去下载

![77319398564](imgs/1773193985649.png)

下载之后进行解压，我们就可以找到对应的文件，`.en`、 `.zh`结尾的文件是可以用记事本打开的文本文件，他们就是英文、中文的翻译原文。

![77319430902](imgs/1773194309024.png)

下载后最好改名，方便后续脚本操作



**2.analyze_corpus.py**

脚本作用：用于检测两个翻译原文文件，是否存在，是否行数匹配

脚本源码：

```python
import os


def analyze_corpus(ch_path, en_path):
    """
    分析双语语料库文件的详细信息
    Args:
        ch_path: 中文文件路径
        en_path: 英文文件路径
    """
    # 检查文件是否存在
    if not os.path.exists(ch_path):
        print(f"中文文件不存在: {ch_path}")
        return
    if not os.path.exists(en_path):
        print(f"英文文件不存在: {en_path}")
        return

    # 获取文件大小
    ch_size = os.path.getsize(ch_path)
    en_size = os.path.getsize(en_path)

    # 读取文件内容并统计行数
    with open(ch_path, 'r', encoding='utf-8') as f:
        ch_lines = f.readlines()
    with open(en_path, 'r', encoding='utf-8') as f:
        en_lines = f.readlines()

    # 计算字符数（不包括换行符）
    ch_chars = sum(len(line.strip()) for line in ch_lines)
    en_chars = sum(len(line.strip()) for line in en_lines)

    # 打印统计信息
    print("=" * 50)
    print("语料库统计信息")
    print("=" * 50)
    print(f"\n中文文件 ({ch_path}):")
    print(f"  文件大小: {ch_size / 1024 / 1024:.2f} MB")
    print(f"  总行数: {len(ch_lines):,}")
    print(f"  总字符数: {ch_chars:,}")
    print(f"  平均每行字符数: {ch_chars / len(ch_lines):.1f}")

    print(f"\n英文文件 ({en_path}):")
    print(f"  文件大小: {en_size / 1024 / 1024:.2f} MB")
    print(f"  总行数: {len(en_lines):,}")
    print(f"  总字符数: {en_chars:,}")
    print(f"  平均每行字符数: {en_chars / len(en_lines):.1f}")

    # 验证中英文行数是否匹配
    if len(ch_lines) == len(en_lines):
        print("\n✓ 中英文文件行数匹配")
    else:
        print("\n✗ 警告：中英文文件行数不匹配！")

    print("=" * 50)


if __name__ == "__main__":
    ch_path = 'corpus.ch'
    en_path = 'corpus.en'
    analyze_corpus(ch_path, en_path)

```



输出效果：

![77319533534](imgs/1773195335349.png)



**3.get_train_dev_test.py**

脚本作用：将corpus.ch、corpus.en原文每行一一对应地放入json文件中，并划分出训练集`train.json`、开发集`dev.json`、验证集`test.json`

脚本原文：

```python
import json
import random

def split_corpus(zh_file, en_file, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1, shuffle=True, seed=42):
    """
    将对齐的中英文语料文件分割成 train/dev/test 并保存为 JSON 格式。

    参数:
        zh_file: 中文文件路径，每行一个句子
        en_file: 英文文件路径，每行一个句子
        train_ratio: 训练集比例
        dev_ratio: 开发集比例
        test_ratio: 测试集比例（需满足三者之和为1）
        shuffle: 是否随机打乱数据
        seed: 随机种子，用于复现
    """
    # 检查比例之和是否为1
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-9, "比例之和必须为1"

    # 读取中文和英文文件，去除行尾换行符
    with open(zh_file, 'r', encoding='utf-8') as f:
        zh_lines = [line.strip() for line in f]
    with open(en_file, 'r', encoding='utf-8') as f:
        en_lines = [line.strip() for line in f]

    # 检查行数是否一致
    assert len(zh_lines) == len(en_lines), f"中英文文件行数不一致：{len(zh_lines)} vs {len(en_lines)}"

    # 组成句对列表
    pairs = [[en, zh] for en, zh in zip(en_lines, zh_lines)]

    # 随机打乱
    if shuffle:
        random.seed(seed)
        random.shuffle(pairs)

    # 计算分割点
    total = len(pairs)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)

    train_pairs = pairs[:train_end]
    dev_pairs = pairs[train_end:dev_end]
    test_pairs = pairs[dev_end:]

    # 写入 JSON 文件
    with open('./json/train.json', 'w', encoding='utf-8') as f:
        json.dump(train_pairs, f, ensure_ascii=False, indent=2)
    with open('./json/dev.json', 'w', encoding='utf-8') as f:
        json.dump(dev_pairs, f, ensure_ascii=False, indent=2)
    with open('./json/test.json', 'w', encoding='utf-8') as f:
        json.dump(test_pairs, f, ensure_ascii=False, indent=2)

    # 打印统计信息
    print(f"总句对数: {total}")
    print(f"训练集: {len(train_pairs)} 句对 ({len(train_pairs)/total:.1%})")
    print(f"开发集: {len(dev_pairs)} 句对 ({len(dev_pairs)/total:.1%})")
    print(f"测试集: {len(test_pairs)} 句对 ({len(test_pairs)/total:.1%})")
    print("已保存为 train.json, dev.json, test.json")

if __name__ == "__main__":
    # 使用示例：假设语料文件名为 corpus.zh 和 corpus.en
    split_corpus('corpus.ch', 'corpus.en', train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1)
```

输出：

![77319645536](imgs/1773196455362.png)

划分效果`以./json/train.json`为例子





**4.`./json`**

`./json`下面存放着经过脚本**get_train_dev_test.py**，划分好的训练集、开发集、测试集





















