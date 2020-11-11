# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 17:57:29 2020

@author: Anuj
"""

import math
import random
import numpy as np
import pandas as pd
import nltk
import codecs

types_of_encoding = ["utf8", "cp1252"]
filename = "en_US.twitter.txt"

for encoding_type in types_of_encoding:
    with codecs.open(filename, encoding = encoding_type, errors ='replace') as csvfile:
        data = csvfile.read()

print("Data type:", type(data))
print("Number of letters:", len(data))
print("First 300 letters of the data")
print("-------")
print(data[0:300])
print("-------")

print("Last 300 letters of the data")
print("-------")
print(data[-300:])
print("-------")



def split_to_sentences(data):
    """
    Split data by linebreak "\n"
    
    Args:
        data: str
    
    Returns:
        A list of sentences
    """
    sentences = data.split("\n")
    
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if len(s) > 0]
    
    return sentences  



def tokenize_sentences(sentences):
    """
    Tokenize sentences into tokens (words)
    
    Args:
        sentences: List of strings
    
    Returns:
        List of lists of tokens
    """
    
    # Initialize the list of lists of tokenized sentences
    tokenized_sentences = []
    
    for sentence in sentences:
        
        # Convert to lowercase letters
        sentence = sentence.lower()
        
        # Convert into a list of words
        
        tokenized = nltk.word_tokenize(sentence)
        
        # append the list of words to the list of lists
        tokenized_sentences.append(tokenized)
        
    return tokenized_sentences 



def get_tokenized_data(data):
    """
    Make a list of tokenized sentences
    
    Args:
        data: String
    
    Returns:
        List of lists of tokens
    """
    
    # Get the sentences by splitting up the data
    sentences = split_to_sentences(data)
    
    # Get the list of lists of tokens by tokenizing the sentences
    tokenized_sentences = tokenize_sentences(sentences)
    
    
    return tokenized_sentences

# Tokenizing and splitting into train and test data
tokenized_data = get_tokenized_data(data)
print(tokenized_data[:4])
random.seed(87)
random.shuffle(tokenized_data)

train_size = int(len(tokenized_data) * 0.8)
train_data = tokenized_data[0:train_size]
test_data = tokenized_data[train_size:]


print("{} data are split into {} train and {} test set".format(
    len(tokenized_data), len(train_data), len(test_data)))

print("First training sample:")
print(train_data[0])
      
print("First test sample")
print(test_data[0])



def count_words(tokenized_sentences):
    """
    Count the number of word appearence in the tokenized sentences
    
    Args:
        tokenized_sentences: List of lists of strings
    
    Returns:
        dict that maps word (str) to the frequency (int)
    """
        
    word_counts = {}
    
    # Loop through each sentence
    for sentence in tokenized_sentences: # complete this line
        
        for token in sentence: # complete this line

            # If the token is not in the dictionary yet, set the count to 1
            if token not in word_counts.keys(): # complete this line
                word_counts[token] = 1
            
            # If the token is already in the dictionary, increment the count by 1
            else:
                word_counts[token] += 1
    
    return word_counts


tokenized_sentences = tokenized_data
word_counts = count_words(tokenized_sentences)


def get_words_with_nplus_frequency(tokenized_sentences, count_threshold):
    """
    Find the words that appear N times or more
    
    Args:
        tokenized_sentences: List of lists of sentences
        count_threshold: minimum number of occurrences for a word to be in the closed vocabulary.
    
    Returns:
        List of words that appear N times or more
    """

    closed_vocab = []
    

    word_counts = count_words(tokenized_sentences)
    

    for word, cnt in word_counts.items(): # complete this line
        

        if cnt >= count_threshold:
            
            # append the word to the list
            closed_vocab.append(word)
    
    return closed_vocab


def replace_oov_words_by_unk(tokenized_sentences, vocabulary, unknown_token="<unk>"):
    """
    Replace words not in the given vocabulary with '<unk>' token.
    
    Args:
        tokenized_sentences: List of lists of strings
        vocabulary: List of strings that we will use
        unknown_token: A string representing unknown (out-of-vocabulary) words
    
    Returns:
        List of lists of strings, with words not in the vocabulary replaced
    """
    
    # Place vocabulary into a set for faster search
    vocabulary = set(vocabulary)
    
    # Initialize a list that will hold the sentences
    # after less frequent words are replaced by the unknown token
    replaced_tokenized_sentences = []
    
    for sentence in tokenized_sentences:
        

        replaced_sentence = []

        for token in sentence: 
            
            if token in vocabulary:
                replaced_sentence.append(token)
            else:
                replaced_sentence.append(unknown_token)
        
        # Append the list of tokens to the list of lists
        replaced_tokenized_sentences.append(replaced_sentence)
        
    return replaced_tokenized_sentences

def preprocess_data(train_data, test_data, count_threshold):
    """
    Preprocess data, i.e.,
        - Find tokens that appear at least N times in the training data.
        - Replace tokens that appear less than N times by "<unk>" both for training and test data.        
    Args:
        train_data, test_data: List of lists of strings.
        count_threshold: Words whose count is less than this are 
                      treated as unknown.
    
    Returns:
        Tuple of
        - training data with low frequent words replaced by "<unk>"
        - test data with low frequent words replaced by "<unk>"
        - vocabulary of words that appear n times or more in the training data
    """

    vocabulary = get_words_with_nplus_frequency(train_data, count_threshold)
    
    train_data_replaced = replace_oov_words_by_unk(train_data, vocabulary, unknown_token="<unk>")
    
    test_data_replaced = replace_oov_words_by_unk(test_data, vocabulary, unknown_token="<unk>")
    
    return train_data_replaced, test_data_replaced, vocabulary



minimum_freq = 2
train_data_processed, test_data_processed, vocabulary = preprocess_data(train_data, 
                                                                        test_data, 
                                                                        minimum_freq)



print("First preprocessed training sample:")
print(train_data_processed[0])
print()
print("First preprocessed test sample:")
print(test_data_processed[0])
print()
print("First 10 vocabulary:")
print(vocabulary[0:10])
print()
print("Size of vocabulary:", len(vocabulary))

def count_n_grams(data, n, start_token='<s>', end_token = '<e>'):
    """
    Count all n-grams in the data
    
    Args:
        data: List of lists of words
        n: number of words in a sequence
    
    Returns:
        A dictionary that maps a tuple of n-words to its frequency
    """
    
    # Initialize dictionary of n-grams and their counts
    n_grams = {}

    
    for sentence in data: # complete this line
        
        # prepend start token n times, and  append <e> one time
        sentence = [start_token]*n + sentence + [end_token]
        
        # convert list to tuple
        # So that the sequence of words can be used as
        # a key in the dictionary
        sentence = tuple(sentence)

        
        for i in range(len(sentence)+1-n): # complete this line

            # Get the n-gram from i to i+n
            n_gram = sentence[i:i+n]

            # check if the n-gram is in the dictionary
            if n_gram in n_grams: 
            
                # Increment the count for this n-gram
                n_grams[n_gram] += 1
            else:
                # Initialize this n-gram count to 1
                n_grams[n_gram] = 1
    
    return n_grams


def estimate_probability(word, previous_n_gram, 
                         n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    """
    Estimate the probabilities of a next word using the n-gram counts with k-smoothing
    
    Args:
        word: next word
        previous_n_gram: A sequence of words of length n
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary_size: number of words in the vocabulary
        k: positive constant, smoothing parameter
    
    Returns:
        A probability
    """
    # Note : 1 . Here we are actually not considering the end token or start token as a part of a vocabulary.
    #        2 . Although the  literature says we need to prepend the n-1 SOS tokens but in reality we are prepending n SOS tokens
    
    # convert list to tuple to use it as a dictionary key
    previous_n_gram = tuple(previous_n_gram)

    previous_n_gram_count = n_gram_counts.get(previous_n_gram, 0)
            

    denominator = float(previous_n_gram_count + (k*vocabulary_size))

    n_plus1_gram = previous_n_gram + (word,)
  

    n_plus1_gram_count = n_plus1_gram_counts.get(n_plus1_gram, 0)
        

    numerator = float(n_plus1_gram_count + k)

    probability = float(numerator/denominator)
    
    
    return probability



unigram_counts = count_n_grams(train_data_processed, 1)
#print(unigram_counts)
bigram_counts = count_n_grams(train_data_processed, 2)
#print(bigram_counts)
tmp_prob = estimate_probability("cat", "a", unigram_counts, bigram_counts, len(vocabulary), k=1)

print(f"The estimated probability of word 'cat' given the previous n-gram 'a' is: {tmp_prob:.4f}")

def estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary, k=1.0):
    """
    Estimate the probabilities of next words using the n-gram counts with k-smoothing
    
    Args:
        previous_n_gram: A sequence of words of length n
        n_gram_counts: Dictionary of counts of n-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary: List of words
        k: positive constant, smoothing parameter
    
    Returns:
        A dictionary mapping from next words to the probability.
    """
    
    previous_n_gram = tuple(previous_n_gram)
    
    # add <e> <unk> to the vocabulary
    # <s> is not needed since it should not appear as the next word
    vocabulary = vocabulary + ["<e>", "<unk>"]
    vocabulary_size = len(vocabulary)
    
    probabilities = {}
    for word in vocabulary:
        probability = estimate_probability(word, previous_n_gram, 
                                           n_gram_counts, n_plus1_gram_counts, 
                                           vocabulary_size, k=k)
        probabilities[word] = probability

    return probabilities

sentences = train_data_processed

estimate_probabilities("a", unigram_counts, bigram_counts, vocabulary, k=1)



def calculate_perplexity(sentence, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0):
    """
    Calculate perplexity for a list of sentences
    
    Args:
        sentence: List of strings
        n_gram_counts: Dictionary of counts of (n+1)-grams
        n_plus1_gram_counts: Dictionary of counts of (n+1)-grams
        vocabulary_size: number of unique words in the vocabulary
        k: Positive smoothing constant
    
    Returns:
        Perplexity score
    """
    # length of previous words
    n = len(list(n_gram_counts.keys())[0]) 
    
    # prepend <s> and append <e>
    sentence = ["<s>"] * n + sentence + ["<e>"]
    
    # Cast the sentence from a list to a tuple
    sentence = tuple(sentence)
    
    # length of sentence (after adding <s> and <e> tokens)
    N = len(sentence)
    

    product_pi = 1.0
    
    
    # Index t ranges from n to N - 1, inclusive on both ends
    for t in range(n, N): 

        # get the n-gram preceding the word at position t
        n_gram = sentence[t-n:t]
        
        # get the word at position t
        word = sentence[t]
        

        probability = estimate_probability(word, n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary_size, k=1.0)
        

        product_pi *= 1/probability

    # Take the Nth root of the product
    perplexity = product_pi**(1/float(N))
    return perplexity






#test_sentence = ['i', 'like', 'a', 'dog']
#test_sentence = test_data_processed[0]
test_sentence = ['thats', 'no', 'fun']
#test_sentence = 'I am doing great'.split(" ")
print(test_sentence)
perplexity_test = calculate_perplexity(test_sentence,
                                       unigram_counts, bigram_counts,
                                       len(vocabulary), k=1.0)
print(f"Perplexity for test sample: {perplexity_test:.4f}")

