import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import sequence
import numpy as np
import pandas as pd
import config
import logging
import os

logging.basicConfig(format='%(asctime)s: %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class SA:
    def __init__(self):
        self.word_index = self.init_word_index()
        self.positive_cap = config.POSITIVE_CAP
        self.neutral_cap = config.NEUTRAL_CAP

        physical_devices = tf.config.list_physical_devices(config.DEVICE)
        try:
            tf.config.experimental.set_memory_growth(
                physical_devices[0], enable=True)
        except Exception as e:
            logging.info(" " + e)

    def init_word_index(self) -> dict:
        word_index = pd.read_csv(config.WORD_INDEX_PATH)
        word_index = dict(zip(word_index.Words, word_index.Indexes))
        word_index["<PAD>"] = 0
        word_index["<START"] = 1
        word_index["<UNK>"] = 2
        word_index["<UNUSED>"] = 3

        return word_index

    def encode_review(self, review: list) -> list:
        tokens = keras.preprocessing.text.text_to_word_sequence(
            " ".join(review))
        tokens = [self.word_index[word]
                  if word in self.word_index else 0 for word in tokens]
        return np.array(tokens)

    def encode_sentiment(self, x: str) -> int:
        if x.lower() == 'positive':
            return 1
        else:
            return 0

    def preprocess_data(self, file_path: str) -> tuple:
        reviews = pd.read_csv(file_path)

        data, labels = reviews['Reviews'], reviews['Sentiment']
        data = data.apply(lambda review: review.split())
        data = data.apply(self.encode_review)

        data = sequence.pad_sequences(data,
                                      value=self.word_index["<PAD>"],
                                      padding='post',
                                      maxlen=500
                                      )
        labels = labels.apply(self.encode_sentiment)

        return (np.array(data), np.array(labels))

    def train_model(self, save_as: str = None):
        train_data, train_labels = self.preprocess_data(config.TRAIN_FILE_PATH)
        test_data, test_labels = self.preprocess_data(config.TEST_FILE_PATH)

        self.model = keras.Sequential([keras.layers.Embedding(10000, 16, input_length=500),
                                       keras.layers.GlobalAveragePooling1D(),
                                       keras.layers.Dense(
                                           16, activation='relu'),
                                       keras.layers.Dense(1, activation='sigmoid')])

        self.model.compile(
            optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.model.fit(train_data, train_labels, epochs=30,
                       batch_size=512, validation_data=(test_data, test_labels))

        self.loss, self.accuracy = self.model.evaluate(test_data, test_labels)

        if save_as == None:
            curr_idx = 0
            for i in os.listdir("./models"):
                if i.startswith("my_model"):
                    idx = int(i[9:])
                    curr_idx = (idx + 1) if curr_idx <= idx else curr_idx
            save_as = f"./models/my_model_{curr_idx}"

        self.model.save(save_as)

    def load_saved_model(self, load_path: str = None) -> None:
        test_data, test_labels = self.preprocess_data(config.TEST_FILE_PATH)

        try:
            self.model = keras.models.load_model(load_path)
        except Exception as e:
            logger.fatal(e)
            raise(e)

        self.loss, self.accuracy = self.model.evaluate(test_data, test_labels)

    def predict(self, review: str) -> tuple:
        review_list = np.array(review.split())
        review_list = self.encode_review(review_list)
        review_list = review_list.reshape(1, review_list.shape[0])
        user_review = sequence.pad_sequences(
            review_list, value=self.word_index["<PAD>"], padding='post', maxlen=500)
        prob = self.model.predict(user_review)

        return (prob[0][0], self.which_sentiment(prob[0][0]))

    def model_details(self) -> None:
        print("Summary".center(66, "="))
        print(self.model.summary())
        print(f"Accuracy {self.accuracy*100}%")
        print(f"Loss: {self.loss}")

    def which_sentiment(self, prob: int) -> str:
        if (prob >= self.positive_cap):
            return "Positive review"

        elif (prob >= self.neutral_cap):
            return "Neutral review"

        else:
            return "Negative review"

    def predict_from_file(self, file_path: str) -> np.ndarray:
        if file_path.endswith('.csv'):
            reviews = pd.read_csv(file_path)
            data = reviews['Reviews']
    
            sentiment_collection = np.array([[0, 0, 0]])
            for i in data:
                prob, sentiment = self.predict(i)
                sentiment_collection = np.append(sentiment_collection, [[i, prob, sentiment]], axis=0)
            
            return sentiment_collection[1:]
                
        elif file_path.endswith('.txt'):
            with open(file_path, 'r') as f:
              data = f.readlines()
            data = [j.replace("\n", "") for j in data]
            
            sentiment_collection = np.array([[0, 0, 0]])
            for i in data:
                prob, sentiment = self.predict(i)
                sentiment_collection = np.append(sentiment_collection, [[i, prob, sentiment]], axis=0)

            return sentiment_collection[1:]

        else:
            raise f"Invalid file format '{file_path}'"
