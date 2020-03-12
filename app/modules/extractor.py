import sys
sys.path.append(".")
from app.models.extractor import Extractor
from loguru import logger
import json

class ExtractorModule:
    def __init__(self):
        pass

    def extract(self, text: str):
        try:
            doc = Extractor.extact(text)
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
        except Exception as e:
            logger.error("Error in extracting entities. Error " + str(e))
            return {"status": False, "Error": str(e)}