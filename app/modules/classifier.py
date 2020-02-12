import sys
sys.path.append(".")
from app.models.classifier import TensorFlowClassifier
from app.modules.cosmos import CosmosDB
from typing import List
from loguru import logger
import json

import app.shared.config as cfg

class Classifier:
    def __init__(self):
        pass

    def create_train(self, name: str, belongsto: str, examples: List[str]):
        '''
        create and train tensor flow classifier classifier
        '''
        try:  
            logger.info("Creating a new classifier: " + name)
            classifier = TensorFlowClassifier(name, belongsto)
            logger.info("Training classifier: " + name)
            classifier.inclass = classifier.train(examples)
            logger.info("Saving classifier: " + classifier.name)
            self.save_classifier(classifier)
            return {"status": True, "classifier": {"id": classifier.id, "name": classifier.name}}
        except Exception as e:
            logger.error("Error in create and train classifier. Error " + str(e))
            return {"status": False, "Error": str(e)}

    def save_classifier(self, classifier: TensorFlowClassifier):
        '''
        save classifier intto cosmos db. Store Embeddings in cosmos
        '''
        try:
            logger.info("Establishing DB connection")
            db = CosmosDB(cfg.settings["host"], cfg.settings["master_key"])
            logger.info("Upserting Item")
            upserted_item = db.upsert_item(cfg.settings["database_id"], cfg.settings["collection_id"], classifier.__dict__)
            logger.info("Upserted Item")
            return upserted_item
        except Exception as e: 
            logger.error("Error in saving classifier: " + classifier.name +". Error " + str(e))
            raise

    def get_classifiers(self, classifier_id: str = None, belongs_to: str = None, active: bool = True, inactive: bool = False):
        '''
        Get all classifiers belongs to a user

        By default, returns all active classifiers
        '''
        try:
            query = 'SELECT * FROM c WHERE'
            if active and not inactive:
                query += ' c.active = true'
            if inactive and not active:
                query += ' c.active = false'
            if active and inactive:
                query += ' c.active = true and c.active = false'
            if classifier_id is not None:
                query += (' and c.id = "' + classifier_id + '"')
            if belongs_to is not None:
                query += (' and c.belongsto = "' + belongs_to + '"')
            
            logger.info("Query: " + query)
            logger.info("Establishing DB connection")
            db = CosmosDB(cfg.settings["host"], cfg.settings["master_key"])
            response = db.query_item(cfg.settings["database_id"], cfg.settings["collection_id"], query)
            if response["status"] is True:
                return response
            else:
                return {"status": False, "Error": "Unable to query items"}   
        except Exception as e:
            logger.error("Error in querying query: " + query + ". Error " + str(e))
            return {"status": False, "Error": str(e)}

    def inference_classifier(self, classifier_id: str, text: str):
        '''
        get inference for text from a particular classifier
        '''
        try:
            logger.info("Fetching embeddings for classifier: " + classifier_id)
            response = self.get_classifiers(classifier_id)
            if response["status"] is True:
                classifiers_embeddings = []
                for item in response["items"]:
                    classifiers_embeddings = item["inclass"]["embeddings"]
                    break
                logger.info("Fetching inference for text.")
                inference = TensorFlowClassifier.inference(classifiers_embeddings, text)
                logger.info("Fetched inference for text.")
                return {"status": True, "Inference": inference}    
            else:
                return {"status": False, "Error": "Unable to fetch embeddings of: " + classifier_id}    
        except Exception as e:
            logger.error("Error in Inferencing classifier: " + classifier_id + ". Error " + str(e))
            return {"status": False, "Error": str(e)}
            