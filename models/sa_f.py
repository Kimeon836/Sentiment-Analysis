# %% [markdown]
# # Sentiment analysis of IMDB reviews
# We will start by importing the necessary libraries

# %%

import tensorflow as tf

# %%

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

import tensorflow.keras as keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %% [markdown]
# # Importing the data files
# After importing the necessary libraries now we will read the data files we have two data files here
# 
# 

# %%
imdb_reviews=pd.read_csv("./imdb_reviews.csv")
test_reviews=pd.read_csv("./test_reviews.csv")

# %% [markdown]
# first data file contains the imdb reviews and their corresponding sentiments which can be either positive or negative, we are going to use this file as our training data.

# %%
imdb_reviews.head()

# %% [markdown]
# the second file is also similar to the first file but we are going to use it as the test data.

# %%
test_reviews.head()

# %% [markdown]
# # Preprocessing the data
# We can not pass the string data to our model directly, so we need to transform the string data into integer format.For this we can map each distinct word as a distinct integer for eg.{'this':14 , 'the':1}.We already have a file that contains the mapping from words to integers so we are going to load that file.
# 

# %%
word_index=pd.read_csv("./word_indexes.csv")

# %% [markdown]
# The word index file contains mapping from words to integers.

# %%
word_index.head()

# %% [markdown]
# Next we are going to convert the word_index dataframe into a python dictionary so that we can use it for converting our reviews from string to integer format.

# %%
word_index=dict(zip(word_index.Words,word_index.Indexes))

# %%
word_index["<PAD>"]=0
word_index["<START"]=1
word_index["<UNK>"]=2
word_index["<UNUSED>"]=3

# %% [markdown]
# Now we define a function review_encoder that encodes the reviews into integer format according to the mapping specified by word_index file.

# %%
def review_encoder(text):
  arr=[word_index[word] for word in text]
  return arr

# %% [markdown]
# We split the reviews from their corresponding sentiments so that we can preprocess the reviews and sentiments separately and then later pass it to our model.

# %%
train_data,train_labels=imdb_reviews['Reviews'],imdb_reviews['Sentiment']
test_data, test_labels=test_reviews['Reviews'],test_reviews['Sentiment']

# %% [markdown]
# Before transforming the reviews as integers we need to tokenize or split the review on the basis of whitespaces
# For eg.the string "The movie was wonderful" becomes ["The" , "movie" , "was" , "wonderful" ].

# %%
train_data=train_data.apply(lambda review:review.split())
test_data=test_data.apply(lambda review:review.split())
train_data[0]

# %% [markdown]
# Since we have tokenized the reviews now we can apply the review_encoder function to each review and transform the reviews into integer format.

# %%
train_data=train_data.apply(review_encoder)
test_data=test_data.apply(review_encoder)
test_data[0]

# %% [markdown]
# After transforming, our reviews are going to look like this.

# %%
train_data.head()

# %% [markdown]
# We also need to encode the sentiments and we are labeling the positive sentiment as 1 and negative sentiment as 0.

# %%
def encode_sentiments(x):
  if x=='positive':
    return 1
  else:
    return 0

train_labels=train_labels.apply(encode_sentiments)
test_labels=test_labels.apply(encode_sentiments)

# %% [markdown]
# Before giving the review as an input to the model we need to perform following preprocessing steps:
# 
#  
# 
# 
# *   The length of each review should be made equal for the model to be working correctly.
# 
# *  We have chosen the length of each review to be 500. 
# *     If the review is longer than 500 words we are going to cut the extra part of the review.
# 
# 
# *       If the review is contains less than 500 words we are going to pad the review with zeros to increase its length to 500.
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# %%
print(train_data.shape)
train_data=keras.preprocessing.sequence.pad_sequences(train_data,value=word_index["<PAD>"],padding='post',maxlen=500)
test_data=keras.preprocessing.sequence.pad_sequences(test_data,value=word_index["<PAD>"],padding='post',maxlen=500)
train_data.shape

# %% [markdown]
# #Building the model
# Our model is a neural network and it consits of the following layers : 
# 
# 1.   one word embedding layer which creates word embeddings of length 16 from integer encoded review.
# 2.  second layer is global average pooling layer which is used to prevent overfitting by reducing the number of parameters.
# 
# 1.   then a dense layer which has 16 hidden units and uses relu as activation function
# 2.  the final layer is the output layer which uses sigmoid as activation function 
# 
# 
# 

# %%
model=keras.Sequential([keras.layers.Embedding(10000,16,input_length=500),
                        keras.layers.GlobalAveragePooling1D(),
                        keras.layers.Dense(16,activation='relu'),
                        keras.layers.Dense(1,activation='sigmoid')])

# %% [markdown]
# #compiling the model
# 
# 
# 1.   Adam is used as optimization function for our model.
# 2.   Binary cross entropy loss function is used as loss function for the model.
# 
# 1.   Accuracy is used as the metric for evaluating the model.
# 
# 
# 
# 

# %%
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# %% [markdown]
# In the next step we are going to train the model on our downloaded IMDB dataset.

# %%
#training the model
history=model.fit(train_data,train_labels,epochs=30,batch_size=512,validation_data=(test_data,test_labels))

# %% [markdown]
# Now we will be evaluating the loss and accuracy of our model on testing data.

# %%
loss,accuracy=model.evaluate(test_data,test_labels)

# %% [markdown]
# As we can see our model is giving an accuracy of 88.58% on the testing data.

# %% [markdown]
# Now we are going to take a random review from our test dataset and check wether our model produces correct output or not

# %%
index=np.random.randint(1,1000)
from keras.preprocessing import sequence
user_review=test_reviews.loc[index]
user_review=test_data[index]
user_review=np.array([user_review])
text = "fine"
text = np.array(text.split())
text = np.array(review_encoder(text))
text = text.reshape(1, text.shape[0])
print(user_review.shape)
user_review = sequence.pad_sequences(text, value=word_index["<PAD>"],padding='post',maxlen=500)
a = model.predict(user_review)
print(a)
if (a>0.5).astype("int32"):
  print("positive sentiment")
else:
  print("negative sentiment")


# %% [markdown]
# As we can see the sentiment for the above review is positive, now we are going to take the integer format of this particular review which we already have in our preprocessed test data and then give it as an input to our model to check the prediction of our model.

# %%



# %% [markdown]
# As we can see our model is now able to predict the sentiment of the review.


