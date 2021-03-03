import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

#Pre-processing Libraries and downloads
import nltk 
import re
nltk.download('words')
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stop_words = list(stop_words)

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

from nltk import word_tokenize

import string
print(string.punctuation)
words = set(nltk.corpus.words.words())
words = list(words)

def remove(list): 
    pattern = '[0-9]'
    list = [re.sub(pattern, '', i) for i in list] 
    pattern1 = '[a-z]'
    return list

def main(): 
    #Importing Training and Validation Dataset
    dataset = pd.read_csv('train1.csv')
    # Boilerplate Data
    boilerplate_data = dataset.iloc[:,[2,-1]].values
    text_list = list(boilerplate_data[:,0])
    
    #Importing Test Dataset
    test_ds = pd.read_csv('test.csv')    
    boilerplate_test_data = test_ds.iloc[:,2].values
    test_text_list = list(boilerplate_test_data)  
        
    
    # Train Validation Split
    train, val = train_test_split(boilerplate_data, test_size=0.2)
    
    # Pre-Processing training text
    def cleaning_text(text):
        cleaned_text = []
        len_data = len(text)
        for i in range(0,len_data):
            sentence = text[i]
            sentence = sentence.lower()
            sentence_p = "".join([char for char in sentence if char not in string.punctuation])
            sentence_words = word_tokenize(sentence_p)
            sentence_filtered = [word for word in sentence_words if word not in stop_words]
            sentence_stemmed = [porter.stem(word) for word in sentence_filtered]
            #sentence_processed = list(w for w in sentence_stemmed if w in words)
            #print(sentence_stemmed)
            sentence_stemmed = remove(sentence_stemmed)
            listToStr = ' '.join([str(elem) for elem in sentence_stemmed])
            #print(listToStr)
            cleaned_text.append(listToStr)
        return cleaned_text
      
    train_text= list(train[:,0])
    train_cleaned_text = cleaning_text(train_text)
        
    # Pre-Processing validation text
    val_text= list(val[:,0])
    val_cleaned_text = cleaning_text(val_text)
    
    # Pre-Processing test data
    test_cleaned_text = cleaning_text(test_text_list)
    
    # Tokenization of textual data
    train_targets = list(train[:,1])
    train_targets = np.array(train_targets)
    val_targets = list(val[:,1])
    val_targets = np.array(val_targets)
    
    
    tokenizer = Tokenizer(num_words=20000,oov_token='<OOV>',split=' ')
    tokenizer.fit_on_texts(train_cleaned_text)
    word_index = tokenizer.word_index
    
    train_text_list = []
    train_text_list = tokenizer.texts_to_sequences(train_cleaned_text)
    train_text_list = pad_sequences(train_text_list,padding='pre',truncating='post',maxlen=20)
    
    # Tokenization of Validation data
    val_text_list = tokenizer.texts_to_sequences(val_cleaned_text)
    val_text_list = pad_sequences(val_text_list,padding='pre',truncating='post',maxlen=20)
    
    # Tokenization of Test Data
    test_text_list = tokenizer.texts_to_sequences(test_cleaned_text)
    test_text_list = pad_sequences(test_text_list,padding='pre',truncating='post',maxlen=20)
    
    #Model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(20000, 64,input_length=20),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='mae', optimizer=tf.keras.optimizers.RMSprop(
    learning_rate=0.003, rho=0.9, momentum=0.2, epsilon=1e-07, centered=False,
    name='RMSprop'), metrics=['accuracy','AUC',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
    
    
    NUM_EPOCHS = 25
    history = model.fit(train_text_list,train_targets, epochs=NUM_EPOCHS, validation_data=(val_text_list,val_targets),use_multiprocessing=True)
    
    
    #Result
    results = model.evaluate(val_text_list, val_targets)
    
    # Prediction on Test Data
    results = model.predict(test_text_list)
    results = np.array(results)
    results = np.round(results)
    prediction = pd.DataFrame(results, columns=['predictions']).to_csv('prediction.csv')
    
if __name__ == "__main__":
    main()


