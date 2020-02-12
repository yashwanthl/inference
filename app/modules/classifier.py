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
            return {"status": "success", "classifier": {"id": classifier.id, "name": classifier.name}}
        except Exception as e:
            logger.error("Error in create and train classifier. Error " + str(e))
            return {"status": "Failure", "Error": str(e)}

    def save_classifier(self, classifier: TensorFlowClassifier):
        '''
        save classifier intto cosmos db. Store Embeddings in cosmos
        '''
        try:
            logger.info("Establishing DB connection")
            db = CosmosDB(cfg.settings["host"], cfg.settings["master_key"])
            logger.info("Upserting Item")
            db.upsert_item(cfg.settings["database_id"], cfg.settings["collection_id"], classifier.__dict__)
            logger.info("Upserted Item")
        except Exception as e: 
            logger.error("Error in saving classifier: " + classifier.name +". Error " + str(e))
            raise

    def get_classifier(self, classifier_id: str):
        '''
        get saved classifier with id
        '''
        try:
            query = 'SELECT * FROM c WHERE c.id = "' + classifier_id + '"'
            logger.info("Establishing DB connection")
            db = CosmosDB(cfg.settings["host"], cfg.settings["master_key"])
            response = db.query_item(cfg.settings["database_id"], cfg.settings["collection_id"], query)
            if response["status"] is True:
                for item in response["items"]:
                    return {"status": "success", "item": item}
            else:
                return {"status": "Failure", "Error": "Unable to query items"}
            return response["items"][0]
        except Exception as e:
            logger.error("Error in getting classifier with id: " + classifier_id +". Error " + str(e))
            return {"status": "Failure", "Error": str(e)}