from storage import (
    create_collection,
    ingest_chunks
)

print("Ingesting repository...")

create_collection()

ingest_chunks()

print("Done.")

