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