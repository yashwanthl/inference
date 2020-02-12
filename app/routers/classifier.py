from fastapi import APIRouter
import sys
sys.path.append(".")
from pydantic import BaseModel
from typing import List
from loguru import logger

from app.modules.classifier import Classifier

router = APIRouter()

class ModelRequest(BaseModel):
    name: str
    user: str
    examples: List[str]

@router.get("/")
def get(id: str = None, belongsto: str = None):
    '''
    End point to get classifiers based on the filters

    parameters
    active: Active classifiers
    id: Classifier Id
    belongsto: Belongs to (Created user)
    '''
    classifier = Classifier()
    response = classifier.get_classifiers(id, belongsto)
    return response

@router.post("/")
def create_train(request: ModelRequest):
    '''
    End point to create and train a new classifier
    '''
    classifier = Classifier()
    reponse = classifier.create_train(request.name, request.user, request.examples)
    return reponse    

