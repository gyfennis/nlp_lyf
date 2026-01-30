import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

script_dir = os.path.dirname(__file__)
max_len = 40

def load_data(setpath: str):
    path = os.path.join(script_dir, setpath)
    dataset = pd.read_csv(path, sep='\t', header=None)
    return dataset

def create_word_table(dataset):
    texts = dataset[0].tolist()
    string_labels = dataset[1].tolist()

    label_to_index = {label: i for i, label in enumerate(set(string_labels))}
    numerical_labels = [label_to_index[label] for label in string_labels]

    char_to_index = {'<pad>': 0}
    for text in texts:
        for char in text:
            if char not in char_to_index:
                char_to_index[char] = len(char_to_index)

    index_to_char = {i: char for char, i in char_to_index.items()}
    vocab_size = len(char_to_index)

    return label_to_index, numerical_labels, char_to_index, index_to_char, vocab_size

class CharTextDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))

        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)
        out = self.fc(hidden_state[-1])

        return out

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden_state = self.gru(embedded)
        out = self.fc(hidden_state[-1])
        return out

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, hidden_state = self.rnn(embedded)
        out = self.fc(hidden_state[-1])
        return out

def train_model(model, dataloader, num_epochs=4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if idx % 50 == 0:
                print(f"Batch 个数{idx}, 当前Batch Loss: {loss.item()}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader)}")

    return model

def evaluate_model(model, test_texts, test_labels, char_to_index, max_len):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for text, true_label in zip(test_texts, test_labels):
            indices = [char_to_index.get(char, 0) for char in text[:max_len]]
            indices += [0] * (max_len - len(indices))
            input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

            output = model(input_tensor)
            _, predicted = torch.max(output, 1)

            if predicted.item() == true_label:
                correct += 1

            total += 1

        accuracy = correct / total
        return accuracy

def classify_text(text, model, char_to_index, max_len, index_to_label):
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

def run_model_comparison():
    # 加载数据
    dataset = load_data("../Week01/dataset.csv")
    label_to_index, numerical_labels, char_to_index, index_to_char, vocab_size = create_word_table(dataset)
    index_to_label = {i: label for label, i in label_to_index.items()}

    # 创建数据集
    text_list = dataset[0].tolist()
    dataset_obj = CharTextDataset(text_list, numerical_labels, char_to_index, max_len)
    dataloader = DataLoader(dataset_obj, batch_size=32, shuffle=True)

    # 模型参数
    embedding_dim = 64
    hidden_dim = 128
    output_dim = len(label_to_index)

    #定义测试数据（从原始数据中取出一部分作为测试）
    test_split = int(0.3 * len(dataset))
    test_texts = text_list[:test_split]
    test_labels = numerical_labels[:test_split]

    models = {
        'LSTM': LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim),
        'GRU': GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim),
        'RNN': RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
    }

    results = {}

    for model_name, model in models.items():
        print(f"\n=== 训练 {model_name} 模型 ===")
        trained_model = train_model(model, dataloader, num_epochs=4)

        print(f"保存 {model_name} 模型 。。。")
        torch.save(trained_model.state_dict(), f"{model_name}.pth")

        print(f"评估 {model_name} 模型...")
        accuracy = evaluate_model(trained_model, test_texts, test_labels, char_to_index, max_len)
        results[model_name] = accuracy
        print(f"{model_name} 模型准确率: {accuracy:.4f}")

    # 打印对比结果
    print("\n=== 模型对比结果 ===")
    for model_name, accuracy in results.items():
        print(f"{model_name}: {accuracy:.4f}")

    # 测试示例
    print("\n=== 示例预测 ===")
    test_text1 = "帮我导航到北京"
    test_text2 = "查询明天北京的天气"

    for model_name in ['LSTM', 'GRU', 'RNN']:
        model_path = f"{model_name}.pth"
        model_state = torch.load(model_path, map_location=torch.device('cpu'))

        if model_name == 'LSTM':
            model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
        elif model_name == 'GRU':
            model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
        else:
            model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)

        model.load_state_dict(model_state)
        model.eval()

        pred1 = classify_text(test_text1, model, char_to_index, max_len, index_to_label)
        pred2 = classify_text(test_text2, model, char_to_index, max_len, index_to_label)

        print(f"输入: '{test_text1}' 预测为: {pred1}")
        print(f"输入: '{test_text2}' 预测为: {pred2}")

if __name__ == "__main__":
    run_model_comparison()
