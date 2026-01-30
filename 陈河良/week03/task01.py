import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
# 文本列表
texts = dataset[0].tolist()
# 标签字符串列表
string_labels = dataset[1].tolist()

# 构建一个字典：标签-》序号
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
# 将标签列表转化为序号列表
numerical_labels = [label_to_index[label] for label in string_labels]

# 构建字符词典，遍历所有文本中的每个字符，未见过的字符分配新索引
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

# 构建一个字典：序号-》字符
index_to_char = {i: char for char, i in char_to_index.items()}
# 字符词典大小
vocab_size = len(char_to_index)

# 所有文本统一截断或填充到 40 个字符长度
max_len = 40


# 自定义DataSet类，继承DataSet
class CharDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 从文本列表获取第idx个文本
        text = self.texts[idx]
        # 截取文本前max_len个字符，并转化为索引
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        # 不足40填充至40
        indices += [0] * (self.max_len - len(indices))
        # 返回一个二元组，包含两个元素：indices转化的torch张量（40,），索引为idx的真实标签
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


# --- NEW RNN Model Class ---
# 定义RNN模型
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        # 字符嵌入层：将每个字符索引（0～vocab_size-1）映射为 embedding_dim 维向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 随机编码的过程， 可训练的
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        self.fc = nn.Linear(hidden_dim, output_dim)  # 分类头
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # 字符索引 → 嵌入向量。
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        # 初始化隐藏状态
        out, hidden = self.rnn(embedded)
        last_output = out[:, -1, :]
        logits = self.fc(last_output)
        return logits
    
# --- NEW LSTM Model Class ---
# 定义LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        # 字符嵌入层：将每个字符索引（0～vocab_size-1）映射为 embedding_dim 维向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 随机编码的过程， 可训练的
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 字符索引 → 嵌入向量。
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out
    
# --- NEW GRU Model Class ---
# 定义GRU模型
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        # 词表大小 转换后维度的维度
        # 字符嵌入层：将每个字符索引（0～vocab_size-1）映射为 embedding_dim 维向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 随机编码的过程， 可训练的
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 字符索引 → 嵌入向量。
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        out, hidden = self.gru(embedded)
        last_output = out[:, -1, :]
        logits = self.fc(last_output)
        return logits


# --- Training and Prediction ---
rnn_dataset = CharDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(rnn_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

rnn_model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
lstm_model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
gru_model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)

rnn_criterion = nn.CrossEntropyLoss()
rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)

lstm_criterion = nn.CrossEntropyLoss()
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

gru_criterion = nn.CrossEntropyLoss()
gru_optimizer = optim.Adam(gru_model.parameters(), lr=0.001)

num_epochs = 4
for epoch in range(num_epochs):
    rnn_model.train()
    lstm_model.train()
    gru_model.train()
    
    running_loss_1 = 0.0
    running_loss_2 = 0.0
    running_loss_3 = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        rnn_optimizer.zero_grad()
        rnn_outputs = rnn_model(inputs)
        rnn_loss = rnn_criterion(rnn_outputs, labels)
        rnn_loss.backward()
        rnn_optimizer.step()
        running_loss_1 += rnn_loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch rnn_loss Loss: {rnn_loss.item()}")
        
        lstm_optimizer.zero_grad()
        lstm_outputs = lstm_model(inputs)
        lstm_loss = lstm_criterion(lstm_outputs, labels)
        lstm_loss.backward()
        lstm_optimizer.step()
        running_loss_2 += lstm_loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch lstm_loss Loss: {lstm_loss.item()}")
        
        gru_optimizer.zero_grad()
        gru_outputs = gru_model(inputs)
        gru_loss = gru_criterion(gru_outputs, labels)
        gru_loss.backward()
        gru_optimizer.step()
        running_loss_3 += gru_loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch gru_loss Loss: {gru_loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], rnn_loss: {running_loss_1 / len(dataloader):.4f}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], lstm_loss: {running_loss_2 / len(dataloader):.4f}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], gru_loss: {running_loss_3 / len(dataloader):.4f}")


def classify_text_rnn(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
rnn_predicted_class = classify_text_rnn(new_text, rnn_model, char_to_index, max_len, index_to_label)
lstm_predicted_class = classify_text_rnn(new_text, lstm_model, char_to_index, max_len, index_to_label)
gru_predicted_class = classify_text_rnn(new_text, gru_model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' rnn预测为: '{rnn_predicted_class}'")
print(f"输入 '{new_text}' lstm预测为: '{lstm_predicted_class}'")
print(f"输入 '{new_text}' gru预测为: '{gru_predicted_class}'")

new_text_2 = "查询明天北京的天气"
rnn_predicted_class_2 = classify_text_rnn(new_text_2, rnn_model, char_to_index, max_len, index_to_label)
lstm_predicted_class_2 = classify_text_rnn(new_text_2, lstm_model, char_to_index, max_len, index_to_label)
gru_predicted_class_2 = classify_text_rnn(new_text_2, gru_model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' rnn预测为: '{rnn_predicted_class_2}'")
print(f"输入 '{new_text_2}' lstm预测为: '{lstm_predicted_class_2}'")
print(f"输入 '{new_text_2}' gru预测为: '{gru_predicted_class_2}'")

