import sys
sys.path.append(".")
# from app.models.flashtext import KeywordProcessor
from flashtext import KeywordProcessor
import pickle
import os
from loguru import logger
from typing import List

class FlaskTextProcessor:
    def __init__(self, name: str):
        self.name = name
        self.processor = None

    def train(self, label_name: str, key_words: List[str]):
        self.processor = KeywordProcessor()
        keyword_dict = {
            label_name: key_words
        }
        self.processor.add_keywords_from_dict(keyword_dict)
        self.serialize()

    def extract(self, text: str):
        try:
            logger.info("Extracting key words")
            # return self.processor.extract_keywords(text, span_info=True)
            prp = FlaskTextProcessor.deserialize(self.name)
            return prp.processor.extract_keywords(text, span_info=True)
        except:
            raise

    def serialize(self):
        '''
        Save the keyword processor
        '''
        output_dir = 'app/data/savedmodels/flashtext/'
        if not os.path.isdir(output_dir):
            logger.info("creating new directory at path: " + output_dir)
            os.makedirs(output_dir)
        logger.info("Saving Model " + self.name + " to directory " + output_dir)
        fname = output_dir + self.name
        with open(fname, 'wb') as f:
            pickle.dump(self.processor, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def deserialize(fname):
        '''
        Return the key word processor with name passed in

        fname: key word processor name

        :return: key word processor

        '''
        keyword_processor = FlaskTextProcessor(fname)
        output_dir = 'app/data/savedmodels/flashtext/'
        logger.info("Reading Key wod processor " + fname + " from directory " + output_dir)
        fname = output_dir + fname
        with open(fname, 'rb') as f:
            keyword_processor.processor = pickle.load(f)

            logger.info("Returing key word processor " + fname)
            print("[INFO]: Returing key word processor " + fname)
            return keyword_processor



    