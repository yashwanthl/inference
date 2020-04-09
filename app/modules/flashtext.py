import sys
sys.path.append(".")
from app.models.flashtextprocessor import FlaskTextProcessor
from typing import List
from loguru import logger

class FlaskTextModule:
    def __init__(self):
        pass

    def create_train(self, fname:str, label_name: str, key_words: List[str]):
        try:
            if (len(key_words) > 0):
                logger.info("Creating a new keyword processor: " + fname)
                flashTextProcessor = FlaskTextProcessor(fname)
                flashTextProcessor.train(label_name, key_words)
                return {"status": True, "Message": "key word processor created successfully"}
            else:
                logger.error("No examples given for key word processor")
                return {"status": False, "Error": "No examples given for key word processor"}
        except Exception as e:
            logger.error("Error in create and train key word processor. Error " + str(e))
            return {"status": False, "Error": str(e)}

    def extract(self, fname: str, text: str):
        try:
            processor = FlaskTextProcessor(fname)
            logger.info("Extracting key words from " + fname + " key word processor")
            entities = []
            keywords = processor.extract(text)
            if (len(keywords) > 0):
                for kw in keywords:
                    eachEntity = {
                        "text": text[kw[1]:kw[2]],
                        "start_char": kw[1],
                        "end_char": kw[2],
                        "label": kw[0]
                    }
                    entities.append(eachEntity)

            return {"status": True, "Entities": entities}
        except Exception as e:
            logger.error("Error in extracting key words. Error " + str(e))
            return {"status": False, "Error": str(e)}