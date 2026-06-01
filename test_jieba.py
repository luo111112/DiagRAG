"""jieba 分词效果测试脚本。"""

import jieba

# 测试语料：医学领域常见文本
TEST_CASES = [
    # 疾病名称
    "急性心肌梗死的诊断标准是什么",
    "慢性阻塞性肺疾病的临床表现",
    "冠状动脉粥样硬化性心脏病治疗方案",
    # 混合中英文
    "ECG 检查显示 ST 段抬高，提示急性心肌梗死",
    # 短句
    "糖尿病用什么药",
    "高血压的并发症有哪些",
    # 药品名
    "阿司匹林肠溶片的用法用量",
    # 标点丰富的句子
    "患者出现胸痛、心悸、呼吸困难等症状，疑似心肌梗死！",
    # 英文词
    "CT 检查报告显示肺部阴影，怀疑 COPD",
]

print("=" * 60)
print("jieba 分词效果测试")
print("=" * 60)

for i, text in enumerate(TEST_CASES, 1):
    tokens = jieba.lcut(text)
    visible = [t if t.strip() else "_" for t in tokens]
    print(f"\n[{i}] 原文: {text}")
    print(f"    分词: {' | '.join(visible)}")
    print(f"    词数: {len([t for t in tokens if t.strip()])}  (已过滤空格/标点空 token)")

print("\n" + "=" * 60)
print("对比：简单正则分词（替换前）")
print("=" * 60)

import re

def simple_tokenize(text: str) -> list[str]:
    tokens = re.split(r"[\s，。、！？；：""''【】《》（）\u4e00-\u9fff]+", text)
    return [t.lower().strip() for t in tokens if t.strip()]

for i, text in enumerate(TEST_CASES, 1):
    simple_tokens = simple_tokenize(text)
    jieba_tokens = [t for t in jieba.lcut(text) if t.strip()]
    simple_vis = [t if t.strip() else "_" for t in re.split(r"[\s，。、！？；：""''【】《》（）\u4e00-\u9fff]+", text)]
    jieba_vis = [t if t.strip() else "_" for t in jieba.lcut(text)]
    print(f"\n[{i}] 原文: {text}")
    print(f"    正则: {' | '.join(simple_vis)}  ({len(simple_tokens)} 词)")
    print(f"    jieba: {' | '.join(jieba_vis)}  ({len(jieba_tokens)} 词)")
