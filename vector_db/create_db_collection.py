from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType

def build_db_collection(uri: str, collection_name: str) -> dict:
    try:
        milvus_client = MilvusClient(uri=uri)

        if milvus_client.has_collection(collection_name):
            milvus_client.drop_collection(collection_name)

        # Define the schema for the collection, specifying the fields and their data types
        schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=True,
        )

        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="organization", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="survey", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="cycle", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="driver_name", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="summary", datatype=DataType.VARCHAR, max_length=3072)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=3072) # text-embedding-3-large dimension length
        schema.add_field(field_name="created_at", datatype=DataType.INT64)

        # Create the collection with the defined schema
        milvus_client.create_collection(
            collection_name=collection_name,
            schema=schema,
            metric_type="IP",
            consistency_level="Strong",
        )

        # Prepare index parameters for the vector field
        index_params = MilvusClient.prepare_index_params()

        # Add an index to the vector field to optimize search operations
        index_params.add_index(
            field_name="vector",
            metric_type="IP",
            index_type="FLAT",
            index_name="vector_index",
            params={ "nlist": 3072 }
        )

        # Create the index in the collection
        milvus_client.create_index(
            collection_name=collection_name,
            index_params=index_params
        )
        return {
                "success": True,
                "message": f"Collection '{collection_name}' created successfully with index."
                }

    except Exception as e:
        return {
                "success": False,
                "message": f"Failed to create collection '{collection_name}': {str(e)}"
                }

# uri= "/mnt/c/Projects/Milvus_RAG_PydanticAI/milvus_tgps.db"
# collection_name="driver_insights_summary"

# print(build_db_collection(uri=uri, collection_name=collection_name))
