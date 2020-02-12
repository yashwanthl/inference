from typing import List
from uuid import uuid4
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import numpy as np
from pprint import pprint
import json
from loguru import logger

nlp = spacy.load("en_core_web_sm")
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

class TensorFlowClassifier:
    def __init__(self, name: str, belongsto: str):
        self.id = str(uuid4())
        self.name = name
        self.belongsto = belongsto
        self.active = True
        self.inclass = None
        self.outclass = None

    def train(self, example_list: List[str]):
        logger.info("Embedding all texts")
        embeddings = embed(example_list).numpy().tolist()
        logger.info("Embedding Done")
        in_class = {
            "examples": example_list,
            "embeddings": embeddings
        }
        return in_class 