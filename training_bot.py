# data tools
import numpy as np
import json

from nltk_utils import binary_word_bag, tokenization, stemming
from modeling_bot import NeuralNetworkModel

# ML library
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

with open('intents.json', 'r') as intents_file:
    intents = json.load(intents_file)

matrix_XY = []
total_words = []
tags = []


# loops through all sentences in intents.json "patterns"
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenization(pattern)
        total_words.extend(w)
        matrix_XY.append((w, tag))

# removes punctuation
ignore_words = ['?', '.', '!', ';', ',']
total_words = [stemming(w) for w in total_words if w not in ignore_words]

# sorts and removes duplicates
total_words = sorted(set(total_words))
tags = sorted(set(tags))

print(len(matrix_XY), "patterns")
print(len(tags), "tags:", tags)
print(len(total_words), "Unique stemmed words:", total_words)

# training data
# binary list (for every sentence)
X_training = []
# tag position (reference the tags)
Y_training = []

for (pattern_sentence, tag) in matrix_XY:
    # sentence transform into binary word
    bin_bag = binary_word_bag(pattern_sentence, total_words)
    X_training.append(bin_bag)
    # find the tag corresponding to the sentence
    label = tags.index(tag)
    Y_training.append(label)

# training data
X_train = np.array(X_training)
Y_train = np.array(Y_training)

# hyper-parameters for the neuralnet parameter
number_epochs = 1000
learning_rate = 0.001
input_size = len(X_train[0])  # same size for bag of words
output_size = len(tags)
hidden_size = 8
batch_size = 8  # dataset split into 8

print(f"Input size: {input_size}, Output size: {output_size}")


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        # input for binary sentence
        self.x_data = X_train
        # output the tag
        self.y_data = Y_train

    # element is gathered through indexing
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # returns size
    def __len__(self):
        return self.n_samples


# creating dataset
dataset = ChatDataset()
# loading dataset
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# utilize GPU processing if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# creating model
model = NeuralNetworkModel(input_size, hidden_size, output_size).to(device)

# loss and optimizer
# entropy loss (log loss) where 1 = bad prediction and 0 = good prediction
criterion = nn.CrossEntropyLoss()
# optimization
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training neural network model
for epoch in range(number_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # data passed forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # every 100th step
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{number_epochs}], Loss: {loss.item()}')

print(f'Final loss: {loss.item()}')

data = {
    "hidden_size": hidden_size,
    "output_size": output_size,
    "total_words": total_words,
    "model_state": model.state_dict(),
    "input_size": input_size,
    "tags": tags
}

# saving model
FILE = "data.pth"
torch.save(data, FILE)

print(f'\nModel finished training. File will be saved within {FILE}')
