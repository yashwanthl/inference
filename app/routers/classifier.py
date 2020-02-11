from fastapi import APIRouter
import sys
sys.path.append(".")
from pydantic import BaseModel
from typing import List
import json
from loguru import logger

from app.models.classifier import Classifier

router = APIRouter()

class ModelRequest(BaseModel):
    name: str
    user: str
    examples: List[str]

@router.get("/")
def status_spam():
    return {"message": "Classifier is up and running"}

@router.post("/")
def create_train(request: ModelRequest):
    '''
    End point to create and train a new classifier
    '''
    logger.info("Creating a new classifier: " + request.name)
    classifier = Classifier(request.name, request.user)
    logger.info("Training classifier: " + request.name)
    classifier.in_class = classifier.train(request.examples)
    logger.info("Training done")
    print(json.dumps(classifier.__dict__))
    return {
        "id": classifier.id,
        "name": classifier.name
    }

