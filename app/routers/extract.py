from fastapi import APIRouter
import sys
sys.path.append(".")
from loguru import logger
from app.modules.extractor import ExtractorModule

router = APIRouter()

@router.get("/")
def get(text: str):
    '''
    End point to extract labels from a give text 

    parameters
    text: text from which labels has to be extracted
    '''
    extractor =  ExtractorModule()
    response = extractor.extract(text)
    return response