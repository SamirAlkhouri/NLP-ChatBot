# NLP-ChatBot
A deep learning chatbot built in PyTorch which utilizes a conversational model to communicate with the user. The chatbot is trained with natural language processing on preset data to generate human-like responses in a conversational setting. 

# Setup
Install PyTorch (https://pytorch.org/get-started/locally/) 

Create enviroment:
```
install python3.9
virtualenv venv --python=python3.9
pip install nltk
```
Potentiol error on first run ('Resource punkt not found.'), run this in the terminal:
```
import nltk
nltk.download('punkt')
```
# Usage
Run ChatBot in terminal:
```
python training_bot.py
python bot_terminal.py
```
Run ChatBot with GUI:
```
python training_bot.py
python chatbot_app.py
```

intents.json file can be customized. Be sure to follow the format accordingly.

