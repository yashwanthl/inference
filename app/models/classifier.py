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
    
    @staticmethod
    def inference(classifier_embeddings: list, text: str):
        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]
        this_embeddings = embed(sentences).numpy().tolist()
        sims = cosine_similarity(this_embeddings, classifier_embeddings)
        stats = [{
            "min": np.min(sim_arr),
            "max": np.max(sim_arr),
            "mean": np.mean(sim_arr),
            "std": np.std(sim_arr)
        } for sim_arr in sims]
        inference = False
        score = 0.0
        evidence = []
        for idx, stat_dict in enumerate(stats):
            score = np.max([score, stat_dict["max"]])
            if (stat_dict["max"] > 0.5 or stat_dict["min"] > 0.2 or stat_dict["mean"] > 0.3):
                inference = True
                evidence.append(sentences[idx])
        return {
            "status": True,
            "inference": inference,
            "score": score,
            "evidence": evidence
        }
