# data tools
import json
import torch
import random

from nltk_utils import binary_word_bag, tokenization
from modeling_bot import NeuralNetworkModel


with open('intents.json', 'r') as intents_file:
    intents = json.load(intents_file)

# opening trained model
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
total_words = data["total_words"]
tags = data["tags"]
model_state = data["model_state"]

# utilize GPU processing if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loading model
model = NeuralNetworkModel(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

chatbot_name = "Chatty"


def main():
    print("\nHey there! Wanna chat? (type 'quit' to exit)")
    print("-" * 45)
    while True:
        sentence = input('You: ')
        if sentence == "quit":
            break

        # input transformation
        sentence = tokenization(sentence)
        X = binary_word_bag(sentence, total_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)
        output = model(X)

        # prediction
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        # chooses random response
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    print(f"{chatbot_name}: {random.choice(intent['responses'])}")
        else:
            print(f"{chatbot_name}: I do not understand, please try again!")


if __name__ == "__main__":
    main()
