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
def status_spam():
    return {"message": "Classifier is up and running"}

@router.post("/")
def create_train(request: ModelRequest):
    '''
    End point to create and train a new classifier
    '''
    classifier = Classifier()
    reponse = classifier.create_train(request.name, request.user, request.examples)
    return reponse

