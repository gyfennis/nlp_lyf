import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
# BertForSequenceClassification bert 用于 文本分类
# Trainer： 直接实现 正向传播、损失计算、参数更新
# TrainingArguments： 超参数、实验设置

from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np

# 加载和预处理数据
dataset_df = pd.read_csv("./sample.csv", sep=",")
n_samples = len(dataset_df)

# 初始化 LabelEncoder，用于将文本标签转换为数字标签
lbl = LabelEncoder()
# 拟合数据并转换前500个标签，得到数字标签
labels = lbl.fit_transform(dataset_df["category"].values[:n_samples])
# 提取前500个文本内容
texts = list(dataset_df["content"].values[:n_samples])

# 分割数据为训练集和测试集
x_train, x_test, train_labels, test_labels = train_test_split(
    texts,             # 文本数据
    labels,            # 对应的数字标签
    test_size=0.2,     # 测试集比例为20%
    stratify=labels    # 确保训练集和测试集的标签分布一致
)




# 从预训练模型加载分词器和模型
tokenizer = BertTokenizer.from_pretrained('./models/bert-base-chinese', local_files_only=True)
model = BertForSequenceClassification.from_pretrained('./models/bert-base-chinese', num_labels=len(set(labels)), local_files_only=True)

# 使用分词器对训练集和测试集的文本进行编码
# truncation=True：如果文本过长则截断
# padding=True：对齐所有序列长度，填充到最长
# max_length=64：最大序列长度
train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=128)

# 将编码后的数据和标签转换为 Hugging Face `datasets` 库的 Dataset 对象
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],           # 文本的token ID
    'attention_mask': train_encodings['attention_mask'], # 注意力掩码
    'labels': train_labels                               # 对应的标签
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels
})





# 定义用于计算评估指标的函数
def compute_metrics(eval_pred):
    # eval_pred 是一个元组，包含模型预测的 logits 和真实的标签
    logits, labels = eval_pred
    # 找到 logits 中最大值的索引，即预测的类别
    predictions = np.argmax(logits, axis=-1)
    # 计算预测准确率并返回一个字典
    return {'accuracy': (predictions == labels).mean()}

# 配置训练参数
training_args = TrainingArguments(
    output_dir='./results',              # 训练输出目录，用于保存模型和状态
    num_train_epochs=4,                  # 训练的总轮数
    per_device_train_batch_size=16,      # 训练时每个设备（GPU/CPU）的批次大小
    per_device_eval_batch_size=16,       # 评估时每个设备的批次大小
    warmup_steps=500,                    # 学习率预热的步数，有助于稳定训练， step 定义为 一次 正向传播 + 参数更新
    weight_decay=0.01,                   # 权重衰减，用于防止过拟合
    logging_dir='./logs',                # 日志存储目录
    logging_steps=100,                   # 每隔100步记录一次日志
    eval_strategy="epoch",               # 每训练完一个 epoch 进行一次评估
    save_strategy="epoch",               # 每训练完一个 epoch 保存一次模型
    load_best_model_at_end=True,         # 训练结束后加载效果最好的模型
)

# 实例化 Trainer 简化模型训练代码
trainer = Trainer(
    model=model,                         # 要训练的模型
    args=training_args,                  # 训练参数
    train_dataset=train_dataset,         # 训练数据集
    eval_dataset=test_dataset,           # 评估数据集
    compute_metrics=compute_metrics,     # 用于计算评估指标的函数
)

# 深度学习训练过程，数据获取，epoch batch 循环，梯度计算 + 参数更新

# 开始训练模型
trainer.train()
# 在测试集上进行最终评估
trainer.evaluate()

# trainer 是比较简单，适合训练过程比较规范化的模型
# 如果我要定制化训练过程，trainer无法满足


model.eval()  # 关闭训练模式，固定模型参数，避免推理受训练层影响
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动识别GPU/CPU
model.to(device)  # 模型移至对应设备
lbl = lbl  # 复用训练时的LabelEncoder，用于还原文本标签


# test_text = "周渝民新片饰演单亲爸爸，与小小彬合作演绎父子情，影片定档5月开机"
test_text = """
春节宜居指南：细看木纹 分辨实木家具(3)
　　实木贴面家具：
　　市场量不大易忽视。在《木家具通用技术条件》实施之前，没有多少人将实木贴面家具称为实木家具，它是在家具表面可视的范围，包括面、框架结构都是纯实木(比如用不知名的杂木)，只不过在实木之上又贴了实木木皮(比如柚木、水曲柳等昂贵点的实木木皮)。
　　优点：价格便宜
　　仅从外观上看，实木贴面家具与前两种实木家具没有差别，但实际上，相比前两种实木家具，实木贴面家具基材材质的成本最低，所以价格相对应该最为便宜。实木贴面家具的线条最为优美，设计也更为简约时尚，在实木家具中，是性价比最高的产品。
　　缺点：最重工艺
　　实木贴面家具最讲究工艺，如果工艺不精，贴面很容易起泡甚至脱落。消费者在购买家具时，一定要详细问清楚，然后看贴木皮处的工艺是否平整。
"""

encoding = tokenizer(
    test_text,
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt"  # 返回PyTorch张量，适配模型输入
)

input_ids = encoding["input_ids"].to(device)
attention_mask = encoding["attention_mask"].to(device)

with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
logits = outputs.logits  # 获取模型原始输出（shape: [1, num_labels]，num_labels=3）

pred_probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # softmax转概率，转为numpy数组
pred_label_idx = np.argmax(pred_probs)  # 取概率最大的索引（预测数字标签）
pred_label_text = lbl.inverse_transform([pred_label_idx])[0]  # 还原为原始文本标签


print("=" * 50)
print("测试文本：", test_text)
print("=" * 50)
print(f"模型原始输出(logits)：{logits.cpu().numpy()[0]}")
print(f"各类别预测概率：{np.round(pred_probs, 4)}")  # 保留4位小数
print(f"预测数字标签：{pred_label_idx}")
print(f"预测文本标签：{pred_label_text}")
print(f"对应标签概率：{np.round(pred_probs[pred_label_idx], 4) * 100:.2f}%")
for idx, cls in enumerate(lbl.classes_):
    print(f"{cls}：{np.round(pred_probs[idx], 4) * 100:.2f}%")
print("=" * 50)
