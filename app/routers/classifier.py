from fastapi import APIRouter
import sys
sys.path.append(".")
from pydantic import BaseModel
from typing import List
from loguru import logger

from app.modules.classifier import Classifier

router = APIRouter()

class CreateClassifierRequest(BaseModel):
    name: str
    user: str
    examples: List[str]

class UpsertClassifierRequest(BaseModel):
    name: str
    user: str
    active: bool = True
    inclass: List[str] = []
    outclass: List[str] = []

@router.get("/")
def get(id: str = None, belongsto: str = None):
    '''
    End point to get classifiers based on the filters

    parameters
    active: Active classifiers
    id: Classifier Id
    belongsto: Belongs to (Created user)
    '''
    logger.info("Request to Get Classifier(s)")
    classifier = Classifier()
    response = classifier.get_classifiers(id, belongsto)
    if response["status"] is True:
        classifiers = []
        for item in response["items"]:
            thisClassifier = {
                "id": item["id"],
                "name": item["name"],
                "belongsto": item["belongsto"],
                "active": item["active"]
            }
            classifiers.append(thisClassifier)
        return {"status": True, "items": classifiers}
    else:
        return {"status": False, "Error": "Unable to fetch classifiers"}

@router.get("/inference")
def get_inference(id: str, text: str):
    '''
    Endpoint to get text inference from a particular classifier
    '''
    logger.info("Request to Get Inferernce from Classifier: " + id)
    classifier = Classifier()
    response = classifier.inference_classifier(id, text)
    return response

@router.post("/")
def create_train(request: CreateClassifierRequest):
    '''
    End point to create and train a new classifier
    '''
    logger.info("Request to Create a new Classifier: " + request.name)
    classifier = Classifier()
    reponse = classifier.create_train(request.name, request.user, request.examples)
    return reponse    

@router.put("/{classifier_id}")
def upsert_train(classifier_id: str, request: UpsertClassifierRequest):
    '''
    End point to Create/Update New/Existing classifier and Train/Retrain it
    '''
    logger.info("Request to Update Classifier: " + request.name)
    classifier = Classifier()
    response = classifier.update_retrain(classifier_id, request.name, request.user, request.active, request.inclass, request.outclass)
    return response