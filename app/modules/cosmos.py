import azure.cosmos.cosmos_client as cosmos_client
from loguru import logger

class CosmosDB:
    def __init__(self, endpoint: str, key: str):
        self.client = cosmos_client.CosmosClient(endpoint, {'masterKey': key})

    def upsert_item(self, database_id, collection_id, item):
        try:
            logger.info("Upserting item to Database: " + database_id + ", Container: " + collection_id + ".")
            database_link = 'dbs/' + database_id
            collection_link = database_link + '/colls/' + collection_id
            upsert_document = self.client.UpsertItem(collection_link, item)
            return {"status": True, "document": upsert_document}
        except Exception as e:
            logger.error("Error while upserting item. Error " + str(e))
            raise

    def query_item(self, database_id, collection_id, query):
        try:
            logger.info("Querying Database: " + database_id + ", Container: " + collection_id + ". Query: " + query)
            database_link = 'dbs/' + database_id
            collection_link = database_link + '/colls/' + collection_id
            logger.info("Fetching items...")
            items_collection = self.client.QueryItems(collection_link, query, {'enableCrossPartitionQuery': True})
            logger.info("Fetched items")
            return {"status": True, "items": items_collection}
        except Exception as e:
            logger.error("Error while querying. Error " + str(e))
            raise