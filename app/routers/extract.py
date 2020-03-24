from fastapi import APIRouter
import sys
sys.path.append(".")
from loguru import logger
from pydantic import BaseModel
from typing import List
from app.modules.extractor import ExtractorModule

router = APIRouter()

class CreateSpacyRequest(BaseModel):
    name: str
    user: str

class RegexMatchRequest(BaseModel):
    words: List[str]

@router.get("")
def get(text: str, name: str = None):
    '''
    End point to extract labels from a give text 

    parameters
    text: text from which labels has to be extracted
    '''
    extractor =  ExtractorModule()
    response = extractor.extract(text, name)
    return response

@router.post("")
def create_train(request: CreateSpacyRequest):
    '''
    End point to create and train spacy ner
    '''
    logger.info("Request to create, train a spacy ner")
    extractor =  ExtractorModule()
    response = extractor.train_spacy(request.name, request.user)
    return response

@router.get("/regex/match")
def regex_match(request: RegexMatchRequest):
    '''
    End point to match various regex types

    Regex types supported
    Date
    Phone Number
    '''
    logger.info("Request to get regex matches")
    extractor =  ExtractorModule()
    reponse = extractor.regex_match(request.words)
    return reponse