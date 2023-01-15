import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

# may need to run the comment below at least once on your device if 'punkt' not found
#nltk.download('punkt')


stemmer = PorterStemmer()


def tokenization(sentence):
    """
    This function will split the sentence into an array of different words
    tokens will be either words, character, or numbers
    """
    return nltk.word_tokenize(sentence)


def stemming(word):
    """
    This function will find the root form of the word
    """
    return stemmer.stem(word.lower())


def binary_word_bag(tokenized_sentence, total_words):
    """
    This function will return a binary array for a bag of words
    it will return 1 if word exists, 0 if not the case
    """
    tokenized_sentence = [stemming(word) for word in tokenized_sentence]
    bin_bag = np.zeros(len(total_words), dtype=np.float32)
    for index, W in enumerate(total_words):
        if W in tokenized_sentence:
            bin_bag[index] = 1
    return bin_bag
