import spacy
import random
from loguru import logger
from uuid import uuid4
import os

from pathlib import Path

TRAIN_DATA = [('what is the price of polo?', {'entities': [(21, 25, 'PrdName')]}), 
              ('what is the price of ball?', {'entities': [(21, 25, 'PrdName')]}), 
              ('what is the price of jegging?', {'entities': [(21, 28, 'PrdName')]}), 
              ('what is the price of t-shirt?', {'entities': [(21, 28, 'PrdName')]}), 
              ('what is the price of jeans?', {'entities': [(21, 26, 'PrdName')]}), 
              ('what is the price of bat?', {'entities': [(21, 24, 'PrdName')]}), 
              ('what is the price of shirt?', {'entities': [(21, 26, 'PrdName')]}), 
              ('what is the price of bag?', {'entities': [(21, 24, 'PrdName')]}), 
              ('what is the price of cup?', {'entities': [(21, 24, 'PrdName')]}), 
              ('what is the price of jug?', {'entities': [(21, 24, 'PrdName')]}), 
              ('what is the price of plate?', {'entities': [(21, 26, 'PrdName')]}), 
              ('what is the price of glass?', {'entities': [(21, 26, 'PrdName')]}), 
              ('what is the price of moniter?', {'entities': [(21, 28, 'PrdName')]}), 
              ('what is the price of desktop?', {'entities': [(21, 28, 'PrdName')]}), 
              ('what is the price of bottle?', {'entities': [(21, 27, 'PrdName')]}), 
              ('what is the price of mouse?', {'entities': [(21, 26, 'PrdName')]}), 
              ('what is the price of keyboad?', {'entities': [(21, 28, 'PrdName')]}), 
              ('what is the price of chair?', {'entities': [(21, 26, 'PrdName')]}), 
              ('what is the price of table?', {'entities': [(21, 26, 'PrdName')]}), 
              ('what is the price of watch?', {'entities': [(21, 26, 'PrdName')]})]

class Extractor:
    def __init__(self, name: str, belongsto: str):
        self.id = str(uuid4())
        self.name = name
        self.belongsto = belongsto
        self.active = True

    def train_spacy(self, data = TRAIN_DATA, iterations = 20):
        nlp = spacy.blank('en')
        if 'ner' not in nlp.pipe_names:
            ner = nlp.create_pipe('ner')
            nlp.add_pipe(ner, last=True)

        for _, annotations in TRAIN_DATA:
            for ent in annotations.get('entities'):
                ner.add_label(ent[2])
        
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):  # only train NER
            optimizer = nlp.begin_training()
            for itn in range(iterations):
                print("Statring iteration " + str(itn))
                random.shuffle(TRAIN_DATA)
                losses = {}
                for text, annotations in TRAIN_DATA:
                    nlp.update(
                        [text],  # batch of texts
                        [annotations],  # batch of annotations
                        drop=0.2,  # dropout - make it harder to memorise data
                        sgd=optimizer,  # callable to update weights
                        losses=losses)
        self.save_spacy(nlp, 'app/data/savedmodels/spaCy/ner/' + self.name + '/')

    def save_spacy(self, nlp, output_dir):
        if not os.path.isdir(output_dir):
            logger.info("creating new directory at path: " + output_dir)
            os.makedirs(output_dir)
        nlp.meta['name'] = self.name  # rename model
        logger.info("Saving model to: " + output_dir)
        nlp.to_disk(output_dir)
        logger.info("Saved model to: " + output_dir)
        print("Saved model to", output_dir)

    # def save_spacy(self, nlp, output_dir):
    #     str_output_dir = output_dir
    #     if output_dir is not None:
    #         output_dir = Path(output_dir)
    #     if not output_dir.exists():
    #         logger.info("creating new directory at path: " + str_output_dir)
    #         output_dir.mkdir()
    #     nlp.meta['name'] = self.name  # rename model
    #     logger.info("Saving model to: " + str_output_dir)
    #     nlp.to_disk(output_dir)
    #     logger.info("Saved model to: " + str_output_dir)
    #     print("Saved model to", str_output_dir)

    @staticmethod
    def extact(text: str):
        logger.info("loading core web sm")
        nlp = spacy.load('en_core_web_sm')
        logger.info("extracting entities")
        doc = nlp(text)
        return doc