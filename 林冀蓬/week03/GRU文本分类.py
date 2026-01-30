import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

max_len = 40

script_dir = os.path.dirname(__file__)

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

class CharGRUDataset(Dataset):
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

def gru_train(model, optimizer, loss_fn, num_epochs, dataloader):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if idx % 50 == 0:
                print(f"Batch 个数{idx}, 当前Batch Loss: {loss.item()}")

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "GRUClassifier.pth")

def classify_text_gru(text, model, char_to_index, max_len, index_to_label):
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

if __name__ == "__main__":
    dataset = load_data("../Week01/dataset.csv")
    label_to_index, numerical_labels, char_to_index, index_to_char, vocab_size = create_word_table(dataset)

    index_to_label = {i: label for label, i in enumerate(numerical_labels)}

    embedding_dim = 64
    hidden_dim = 128
    output_dim = len(label_to_index) #+1
    epochs_num = 4

    model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    gru_dataset = CharGRUDataset(dataset[0], numerical_labels, char_to_index, max_len)
    gru_dataloader = DataLoader(gru_dataset, batch_size=32, shuffle=True)

    gru_train(model, optimizer, loss_fn, epochs_num, gru_dataloader)

    model.load_state_dict(torch.load("GRUClassifier.pth"))
    model.eval()

    new_text = "帮我导航到北京"
    predicted_class = classify_text_gru(new_text, model, char_to_index, max_len, index_to_label)
    print(f"输入： '{new_text}' 预测为： '{predicted_class}'")

    new_text_2 = "查询明天北京的天气"
    predicted_class_2 = classify_text_gru(new_text_2, model, char_to_index, max_len, index_to_label)
    print(f"输入： {new_text_2} 预测为： '{predicted_class_2}'")