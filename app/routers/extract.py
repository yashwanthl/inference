from fastapi import APIRouter
import sys
sys.path.append(".")
from loguru import logger
from pydantic import BaseModel
from typing import List
from app.modules.extractor import ExtractorModule
from app.modules.flashtext import FlaskTextModule

router = APIRouter()

class CreateSpacyRequest(BaseModel):
    name: str
    user: str

class RegexMatchRequest(BaseModel):
    words: List[str]

class CreateKeywordProcessorRequest(BaseModel):
    name: str
    label: str
    words: List[str]

@router.get("")
def get(text: str, name: str = None):
    '''
    End point to extract labels from a give text 

    parameters
    text: text from which labels has to be extracted
    '''
    extractor =  ExtractorModule()
    response = extractor.extract_all(text, name)
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

@router.get("/regex/match/date/monthfirst")
def regex_find(text: str):
    '''
    End point to find date matches in a given text
    date formats supported: mm/dd/yyyy or mm-dd-yyyy or mm.dd.yyyy or mmm-dd-yyyy or mmm/dd/yyyy or mmm.dd.yyyy In all the cases year can be yy also 

    PARAMETERS
    ----------
    text: str
    text in which dates to be found if any

    RETURNS
    -------
    Dict like {"status": bool, "Matches": List of all date matches with start and end index}
    '''
    logger.info("Reuqest to find all Month-First date matches in: " + text)
    extractor = ExtractorModule()
    response = extractor.regex_match_date_monthfirst(text)
    return response

@router.post("/flashtext")
def create_keyword_processor(request: CreateKeywordProcessorRequest):
    '''
    End point to label words against  a user defined label
    '''
    logger.info("Request to create a new key word processor")
    flashtext = FlaskTextModule()
    response = flashtext.create_train(request.name, request.label, request.words)
    return response

@router.get("/flashtext")
def extract_keyword(name: str, text: str):
    '''
    end point to extract key words from given flash text processor
    '''
    logger.info("Extracting key words from " + name + " key word processor")
    flashtext = FlaskTextModule()
    response = flashtext.extract(name, text)
    return response
