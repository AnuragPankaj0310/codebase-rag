from storage import qdrant, create_collection, COLLECTION_NAME

print("Deleting collection...")
qdrant.delete_collection(COLLECTION_NAME)

print("Recreating collection...")
create_collection()

print("Done.")