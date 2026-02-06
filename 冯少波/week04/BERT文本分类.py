import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# step1 : 加载数据集，从hugging face上找的，科学分类文本数据集
dataset = load_dataset("knowledgator/Scientific-text-classification")

if 'train' in dataset and 'test' in dataset:
    train_data = dataset['train']
    test_data = dataset['test']
elif 'train' in dataset and 'validation' in dataset:
    train_data = dataset['train']
    test_data = dataset['validation']
elif 'train' in dataset:
    dataset_split = dataset['train'].train_test_split(test_size=0.2,seed=43)
    train_data = dataset_split['train']
    test_data = dataset_split['test']

train_data = train_data.shuffle(seed=42).select(range(min(20000, len(train_data))))
test_data = test_data.shuffle(seed=42).select(range(min(10000, len(test_data))))


# step2 : 处理标签
all_labels = set()
for item in train_data:
    all_labels.add(item['label'])
for item in test_data:
    all_labels.add(item['label'])

label_names = sorted(list(all_labels))
num_labels = len(label_names)

for i, label in enumerate(label_names[:5]):
    print(f"{label}")

label2id = {label: idx for idx, label in enumerate(label_names)}
id2label = {idx: label for idx, label in enumerate(label_names)}

# step 3 数据处理
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize_and_encode_labels(examples):
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=128
    )
    tokenized['labels'] = [label2id[label] for label in examples['label']]
    return tokenized


tokenized_train = train_data.map(
    tokenize_and_encode_labels,
    batched=True,
    remove_columns=['text','label']
)

tokenized_test = test_data.map(
    tokenize_and_encode_labels,
    batched=True,
    remove_columns=['text','label']
)

# step4 : 加载预训练模型
device = torch.device("cpu")
try:
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels = num_labels,
        id2label=id2label,
        label2id=label2id
    )
    model = model.to(device)
except:
    raise

total_params = sum(p.numel() for p in model.parameters())

# step5 定义评估指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

    return {
        'accuracy' : accuracy,
        'precision' : precision,
        'recall' : recall,
        'f1' : f1
    }

# setp 6 : 设置训练参数并训练
training_args = TrainingArguments(
    output_dir='./results_scientific',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    learning_rate=3e-5,
    logging_steps=200,
    eval_strategy="epoch",              # 改为按epoch评估，减少保存次数
    save_strategy="epoch",              # 改为按epoch保存
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=1,                 # 只保留1个checkpoint
    report_to="none",
    fp16=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

trainer.train()

# step 7 : 模型评估 & 保存
eval_results = trainer.evaluate()
for key, value in eval_results.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f} ({value*100:.1f}%)")

model_save_path = './bert_scientific_classifier'
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)


# step 8 : 测试
def predict_text(text, model, tokenizer, label_names):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    inputs = {k:v.to('cpu') for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        probabilities = torch.softmax(logits,dim=-1)

        predicted_class = torch.argmax(probabilities,dim=-1).item()
        confidence = probabilities[0][predicted_class].item()

    return {
        'predicted_label':label_names[predicted_class],
        'predicted_class_id':predicted_class,
        'confidence':confidence,
        'all_probabilities':probabilities[0].cpu().tolist()
    }


test_sample_1 = """
This paper presents a novel deep learning approach for image classification 
using convolutional neural networks. The proposed architecture achieves 
state-of-the-art performance on benchmark datasets including ImageNet and CIFAR-10.
"""

result_1 = predict_text(test_sample_1,model,tokenizer, label_names)
print(f"预测结果：{result_1['predicted_label']}")
print(f"置信度：{result_1['confidence']:.1%}")
print(f"前三个最可能的类别：")
probs_with_labels = [(label_names[i], prob) for i, prob in enumerate(result_1['all_probabilities'])]
probs_with_labels.sort(key=lambda x:x[1], reverse=True)
for i, (label, prob) in enumerate(probs_with_labels[:3],1):
    print(f"{i}.{label:20s}:{prob:.1%}")









