import azure.cosmos.cosmos_client as cosmos_client
from loguru import logger

class CosmosDB:
    def __init__(self, endpoint: str, key: str):
        self.client = cosmos_client.CosmosClient(endpoint, {'masterKey': key})

    def upsert_item(self, database_id, collection_id, item):
        try:
            database_link = 'dbs/' + database_id
            logger.info("Database link: " + database_link)
            collection_link = database_link + '/colls/' + collection_id
            logger.info("Collection link: " + collection_link)
            upsert_document = self.client.UpsertItem(collection_link, item)
            return {"status": True, "document": upsert_document}
        except Exception as e:
            logger.error("Error while upserting item. Error " + str(e))
            raise

    def query_item(self, database_id, collection_id, query):
        try:
            database_link = 'dbs/' + database_id
            logger.info("Database link: " + database_link)
            collection_link = database_link + '/colls/' + collection_id
            logger.info("Collection link: " + collection_link)
            items_collection = self.client.QueryItems(collection_link, query, {'enableCrossPartitionQuery': True})
            return {"status": True, "items": items_collection}
        except Exception as e:
            logger.error("Error while querying. Error " + str(e))
            raise


DATABASE_ID = 'hubbleinference'
COLLECTION_ID = 'classifiers'
END_POINT = 'https://mysqlhubbleinference.documents.azure.com:443/'
KEY = 'WmRmnTHrIEAJlsFArlbS345R8dkKJE2YHzzc8tD22FD9VQ3TqTjExUHLbgdr2torIJo7NnVkuOudXpnTD7iEnw=='
cosmos = CosmosDB(END_POINT, KEY)

database_link = 'dbs/' + DATABASE_ID
collection_link = database_link + '/colls/' + COLLECTION_ID

# # database = cosmos.client.ReadDatabase('dbs/hubbleinference')
# # print('Database with id \'{0}\' was found, it\'s _self is {1}'.format(id, database['_self']))
# # item_to_create = [json.dumps(thisjson)]
# # cosmos.client.CreateItem(collection_link, thisjson)

def upsert_item(item):
    # cosmos.client.CreateItem(collection_link, item)
    cosmos.client.UpsertItem(collection_link, item)

# def delete_item(modelid: str):
#     query = 'SELECT * FROM c WHERE c.id = "'+ modelid +'"'
#     print(query)
#     for item in cosmos.client.QueryItems(collection_link, query, {'enableCrossPartitionQuery': True}):
#         print(item['id'])
#         cosmos.client.DeleteItem(collection_link + "/docs/" + item['id'], {'partitionKey': '/id'})



# insert_item(thisjson)
    
