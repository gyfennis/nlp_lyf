import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time


def load_data():
    df = pd.read_csv("../Week01/dataset.csv", sep="	", header=None)
    df = df.dropna()
    texts = df[0].tolist()
    labels = df[1].tolist()

    label_to_index = {label: i for i, label in enumerate(set(labels))}
    numerical_labels = [label_to_index[l] for l in labels]

    char_to_index = {'<pad>': 0}
    for text in texts:
        for char in str(text):
            if char not in char_to_index:
                char_to_index[char] = len(char_to_index)

    return texts, numerical_labels, char_to_index, label_to_index


class TextDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        indices = [self.char_to_index.get(c, 0) for c in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


class UniversalModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, model_type):
        super(UniversalModel, self).__init__()
        self.model_type = model_type
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if model_type == 'RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        elif model_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        elif model_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        out, hidden = self.rnn(embedded)

        if self.model_type == 'LSTM':
            final_hidden = hidden[0]
        else:
            final_hidden = hidden

        return self.fc(final_hidden.squeeze(0))


def train_and_evaluate(model_type, texts, labels, char_to_index, output_dim):
    vocab_size = len(char_to_index)
    embedding_dim = 64
    hidden_dim = 128
    max_len = 40
    batch_size = 64
    epochs = 5

    dataset = TextDataset(texts, labels, char_to_index, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UniversalModel(vocab_size, embedding_dim, hidden_dim, output_dim, model_type)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()

    final_loss = 0.0
    final_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_preds = 0
        total_preds = 0

        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == targets).sum().item()
            total_preds += targets.size(0)

        final_loss = total_loss / len(dataloader)
        final_acc = correct_preds / total_preds

    end_time = time.time()
    duration = end_time - start_time

    print(f"模型: {model_type} | 耗时: {duration:.2f}s | Loss: {final_loss:.4f} | Acc: {final_acc:.4f}")


if __name__ == "__main__":
    texts, labels, char_map, label_map = load_data()
    output_dim = len(label_map)

    print("开始训练...")
    train_and_evaluate('RNN', texts, labels, char_map, output_dim)
    train_and_evaluate('GRU', texts, labels, char_map, output_dim)
    train_and_evaluate('LSTM', texts, labels, char_map, output_dim)
