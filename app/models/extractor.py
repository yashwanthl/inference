import spacy
from loguru import logger
from uuid import uuid4

class Extractor:
    def __init__(self, name: str, belongsto: str):
        self.id = str(uuid4())
        self.name = name
        self.belongsto = belongsto
        self.active = True

    @staticmethod
    def extact(text: str):
        logger.info("loading core web sm")
        nlp = spacy.load('en_core_web_sm')
        logger.info("extracting entities")
        doc = nlp(text)
        return doc