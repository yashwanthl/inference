import sys
sys.path.append(".")
from app.models.classifier import TensorFlowClassifier
from app.modules.cosmos import CosmosDB
from typing import List
from loguru import logger
from uuid import uuid4
import json
from collections import OrderedDict

import app.shared.config as cfg

class Classifier:
    def __init__(self):
        pass

    def create_train(self, name: str, belongsto: str, examples: List[str]):
        '''
        create and train tensor flow classifier classifier
        '''
        try:
            if (len(examples) > 0):
                logger.info("Creating a new classifier: " + name)
                classifier = TensorFlowClassifier(name, belongsto)
                # Removing Duplicates and Maintaining Order
                logger.info("Removing duplicates if any from training examples")
                examples = list(OrderedDict.fromkeys(examples))
                logger.info("Training classifier: " + name)
                classifier.inclass = classifier.train(examples)
                logger.info("Saving classifier: " + classifier.name)
                created_item = self.save_classifier(classifier)
                if (created_item["status"] and type(created_item["document"]) == dict and created_item["document"]["id"] and created_item["document"]["name"]):
                    logger.info("Classifier with id: " + created_item["document"]["id"] + ", name: " + created_item["document"]["name"] + " created")
                    return {"status": True, "classifier": {"id": created_item["document"]["id"], "name": created_item["document"]["name"]}}
                else:
                    logger.error("Unable to create classifier due to database error")
                    return {"status": False, "Error": "Unable to create classifier due to database error"}
            else:
                logger.error("No training examples to create a classifier")
                return {"status": False, "Error": "No training examples to create a classifier"}
            return {"status": False, "Error": "Unable to create classifier due to unkown errors"}
        except Exception as e:
            logger.error("Error in create and train classifier. Error " + str(e))
            return {"status": False, "Error": str(e)}

    def update_retrain(self, id: str, name: str, belongsto: str, active: bool, inclass: List[str], outclass: List[str]):
        try:
            logger.info("Getting details of: " + name)
            items = []
            response = self.get_classifiers(id)

            if response["status"] is True:
                logger.info("Success in getting details of: " + name)
                for item in response["items"]:
                    thisItem = {
                        "id": item["id"],
                        "name": item["name"],
                        "belongsto": item["belongsto"],
                        "active": item["active"],
                        "inclass": item["inclass"],
                        "outclass": item["outclass"]
                    }
                    items.append(thisItem)
            else:
                logger.error("No existing classifier to update and retrain")
                logger.info("Creating a New classifer as there is no exsisting classfier")
                return self.create_train(name, belongsto, inclass)
            
            if (len(items) > 0):
                existingClassifier = items[0]
                logger.info("Fetching existing inclass examples")
                if (type(existingClassifier["inclass"]) == dict) and (type(existingClassifier["inclass"]["examples"]) == list) and (len(existingClassifier["inclass"]["examples"]) > 0):
                    e_inclass = existingClassifier["inclass"]["examples"]
                    logger.info("Merging existing inclass with new inclass examples")
                    inclass = (e_inclass + inclass)
                    # Removing Duplicates and Maintaining Order
                    logger.info("Removing duplicates in inclass examples")
                    inclass = list(OrderedDict.fromkeys(inclass))
                
                logger.info("Fetching existing outclass examples")
                if (type(existingClassifier["outclass"]) == dict) and (type(existingClassifier["outclass"]["examples"]) == list) and (len(existingClassifier["outclass"]["examples"]) > 0):
                    e_outclass = existingClassifier["outclass"]["examples"]
                    logger.info("Merging existing outclass with new outclass examples")
                    outclass = (e_outclass + outclass)
                    # Removing Duplicates and Maintaining Order
                    logger.info("Removing duplicates in outclass examples")
                    outclass = list(OrderedDict.fromkeys(outclass))

                logger.info("Creating a classifier with existing id")
                classifier = TensorFlowClassifier(name, belongsto)
                classifier.id = id
                classifier.active = active

                if (len(inclass) > 0):
                    logger.info("Retraining classifier: " + name + " with inclass examples")
                    classifier.inclass = classifier.train(inclass)
                if (len(outclass) > 0):
                    logger.info("Retraining classifier: " + name + " with outclass examples")
                    classifier.outclass = classifier.train(outclass)
                
                logger.info("Saving classifier: " + classifier.name)
                created_item = self.save_classifier(classifier)
                if (created_item["status"] and type(created_item["document"]) == dict and created_item["document"]["id"] and created_item["document"]["name"]):
                    logger.info("Classifier with id: " + created_item["document"]["id"] + ", name: " + created_item["document"]["name"] + " updated and retrained")
                    return {"status": True, "classifier": {"id": created_item["document"]["id"], "name": created_item["document"]["name"]}}
                else:
                    logger.error("Unable to update and retrain classifier due to database error")
                    return {"status": False, "Error": "Unable to update and retrain classifier due to database error"}

            else:
                logger.error("No existing classifier to update and retrain")
                logger.info("Creating a New classifer as there is no exsisting classfier")
                return self.create_train(name, belongsto, inclass)
            return {"status": False, "Error": "Unable to retrain classifier due to unkown errors"}
        except Exception as e:
            logger.error("Error in update and retrain classifier. Error " + str(e))
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
            