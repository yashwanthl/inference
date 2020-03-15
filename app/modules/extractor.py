import sys
import ast
sys.path.append(".")
from app.models.extractor import Extractor
from app.modules.dataset import Dataset
from loguru import logger
import json

class ExtractorModule:
    def __init__(self):
        pass

    def extract(self, text: str, name: str = None):
        '''
        Extract entities from spaCy ner model or pre trained spaCy ner model

        parameters
        @text - text from which entities should be extracted
        @name - Optional - name of per trained spaCy ner model
        '''
        try:
            doc = Extractor.extact(text, name)
            if (doc is not None):
                logger.info("Fetched entities")
                logger.info("Building each entity object")
                entities = []
                for ent in doc.ents:
                    thisEnt = {
                        "text": ent.text,
                        "start_char": ent.start_char,
                        "end_char": ent.end_char,
                        "label": ent.label_
                    }
                    logger.info(thisEnt)
                    entities.append(thisEnt)
                return {"status": True, "Entities": entities}
            else:
                return {"status": False, "Error": "Error in fetching entities"}
        except Exception as e:
            logger.error("Error in extracting entities. Error " + str(e))
            return {"status": False, "Error": str(e)}
    
    def train_spacy(self, name: str, belongsTo: str, data = None, iterations = 20):
        '''
        Train spaCy ner

        Parameters
        @data - Training data
        @iterations - default to 20
        '''
        try:
            extractor = Extractor(name, belongsTo)
            success = False
            if data is None:
                ds = Dataset()
                data = ds.get_sampleannotations()
                # ast.literal_eval converts data from a string to list of tuples 
                success = extractor.train_spacy(data = ast.literal_eval(data), iterations = iterations)
            else:
                success = extractor.train_spacy(data = data, iterations = iterations)
            if (success is not None and success):
                return {"status": True, "Message": "Successfully craeted and trained model"}
            else:
                return {"status": False, "Message": "Error in creating and training model"}
        except Exception as e:
            logger.error("Error in creating and training ner. Error " + str(e))
            return {"status": False, "Error": str(e)}