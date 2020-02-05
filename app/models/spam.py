import pickle
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from loguru import logger

class SpamOrHam:
    def __init__(self):
        self.classifier = None
        self.vectorizer = None

    def train(self, emails):
        '''
        Preprocess emails content and train Naive Bayes classifier

        emails: emails dataframe

        '''

        # Check for duplicates and dop those rows
        logger.info("Dropping duplicate records")
        print("[INFO]: Dropping duplicate records")
        emails.drop_duplicates(inplace=True)

        # Tokenization
        logger.info("Tokenization")
        print("[INFO]: Tokenization")
        emails['tokens'] = emails['text'].map(lambda text:  nltk.tokenize.word_tokenize(text))  

        # Removing stop words
        logger.info("Removing stop words")
        print("[INFO]: Removing stop words")
        stop_words = set(nltk.corpus.stopwords.words('english'))
        emails['filtered_text'] = emails['tokens'].map(lambda tokens: [w for w in tokens if not w in stop_words])

        # Removing 'Subject:'
        logger.info("Removing Subject:")
        print("[INFO]: Removing Subject:")
        emails['filtered_text'] = emails['filtered_text'].map(lambda text: text[2:])

        # Mails still have many special charater tokens which may not be relevant for spam filter, lets remove these
        # Joining all tokens together in a string
        logger.info("Removing special characters")
        print("[INFO]: Removing special characters")
        emails['filtered_text'] = emails['filtered_text'].map(lambda text: ' '.join(text))

        # Removing special characters from each mail 
        emails['filtered_text'] = emails['filtered_text'].map(lambda text: re.sub('[^A-Za-z0-9]+', ' ', text))

        # Lemmatization
        logger.info("Lemmatization")
        print("[INFO]: Lemmatization")
        wnl = nltk.WordNetLemmatizer()
        emails['filtered_text'] = emails['filtered_text'].map(lambda text: wnl.lemmatize(text))

        # Bag of Words
        self.vectorizer = CountVectorizer()
        counts = self.vectorizer.fit_transform(emails['filtered_text'].values)
        
        # Naive Bayes Classifier
        logger.info("Naive Bayes Classification")
        print("[INFO]: Naive Bayes Classification")
        self.classifier = MultinomialNB()
        targets = emails['spam'].values
        self.classifier.fit(counts, targets)

    def predict(self, emailText):
        '''
        Predict passed text is spam or not using trained classifier

        emailText: Text to be classified

        :return: list of predictions. 1 is SPAM. 0 is NOT SPAM
        '''
        logger.info("Transforming text")
        print("[INFO]: Transforming text")
        emailTexts_counts = self.vectorizer.transform(emailText)

        logger.info("Predicting")
        print("[INFO]: Predicting...")
        prediction = self.classifier.predict(emailTexts_counts)
        logger.info("Done with Prediction")
        print("[INFO]: Done with Prediction")
        return prediction

    def serialize(self, fname):
        '''
        Save the trained classifer

        fname: Name of the classifier

        '''
        logger.info("Saving Model " + fname)
        print("[INFO]: Saving Model as " + fname)
        with open(fname, 'wb') as f:
            pickle.dump(self.vectorizer, f)
            pickle.dump(self.classifier, f) 

    @staticmethod
    def deserialize(fname):
        '''
        Return the model with name passed in

        fname: model name

        :return: model

        '''
        model = SpamOrHam()
        logger.info("Reading Model " + fname)
        print("[INFO]: Reading Model " + fname)
        try:
            with open(fname, 'rb') as f:
                model.vectorizer = pickle.load(f)
                model.classifier = pickle.load(f)

                logger.info("Returing Model " + fname)
                print("[INFO]: Returing Model " + fname)
                return model
        except Exception as e:
            logger.error("Error Returing Model " + fname + ". Error: " + str(e))
            print("[ERROR]: Returing Model " + fname + ". Error: " + str(e))
            return "Error"