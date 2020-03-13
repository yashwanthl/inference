from fastapi import APIRouter
import sys
sys.path.append(".")
from loguru import logger
from pydantic import BaseModel
from app.modules.extractor import ExtractorModule

router = APIRouter()

class CreateSpacyRequest(BaseModel):
    name: str
    user: str

@router.get("/")
def get(text: str, name: str = None):
    '''
    End point to extract labels from a give text 

    parameters
    text: text from which labels has to be extracted
    '''
    extractor =  ExtractorModule()
    response = extractor.extract(text, name)
    return response

@router.post("/")
def create_train(request: CreateSpacyRequest):
    '''
    End point to create and train spacy ner
    '''
    logger.info("Request to create, train a spacy ner")
    extractor =  ExtractorModule()
    response = extractor.train_spacy(request.name, request.user)
    return response