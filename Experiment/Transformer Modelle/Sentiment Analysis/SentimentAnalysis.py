import pandas as pd
from numpy import array, argmax
import numpy as np

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, TFAutoModel

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-uncased")
bert = TFAutoModel.from_pretrained("dbmdz/bert-base-german-uncased")

max_length=100
path = "/content/gdrive/MyDrive/Experiment/Transformer Modelle/Sentiment Analysis/model/cp.ckpt"

class EmotionModel:
  def __init__(self):
    self.model = self.emotion_model()
    
    
  def emotion_model(self):
    input_ids=tf.keras.layers.Input(shape=(max_length,),name='input_ids',dtype='int32')
    input_mask=tf.keras.layers.Input(shape=(max_length,),name='attention_mask',dtype='int32')
    
    embedding=bert(input_ids,attention_mask=input_mask)[0]
    x=tf.keras.layers.GlobalMaxPool1D()(embedding)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Dense(256,activation='relu')(x)
    x=tf.keras.layers.Dropout(0.2)(x)
    output=tf.keras.layers.Dense(5,activation='softmax')(x)
    
    model=tf.keras.Model(inputs=[input_ids,input_mask],outputs=output)
    
    model.layers[2].trainable=False

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer='adam',metrics=[tf.keras.metrics.AUC()])

    model.load_weights(path)

    return model
    
    
  def predict(self, sequence):
    enc = self.encoding(sequence)
    result = self.model.predict([enc["input_ids"],enc["attention_mask"]])[0]
    
    dict_res = {'anger':result[0],
              'fear':result[1],
              'joy':result[2],
              'neutral':result[3],
              'sadness':result[4]
    }

    return dict_res
    

  def encoding(self, sequence):
    tokens=tokenizer.encode_plus(sequence,max_length=max_length,padding='max_length',add_special_tokens=True,
                        truncation=True,return_token_type_ids=False,return_attention_mask=True,
                        return_tensors='tf')
    
    return tokens


  def neutralize_score(self, sentiment_val, neutral_score):
     # positive sentiment
    if sentiment_val > 0:
        if sentiment_val > neutral_score:
            return sentiment_val - neutral_score
        else:
            return 0
  
     # negative sentiment
    else:
        if abs(sentiment_val) > neutral_score:
            return sentiment_val + neutral_score
        else:
            return 0


  def get_sentiment(self, inp):
    pred = self.predict(inp)
    
    sentiment = sum([-(pred['anger']),-(pred['fear']),-(pred['sadness']),pred['joy']])
    sentiment_val = np.round(sentiment,5)

    if sentiment_val < 0:
        return  "negative"

    elif sentiment_val > 0:
        return  "positive"

    else:
        return  "neutral"


