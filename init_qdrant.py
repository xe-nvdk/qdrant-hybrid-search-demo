from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client.http.models import BinaryQuantization, BinaryQuantizationConfig
from qdrant_client.http.models import PayloadSchemaType
from tqdm import tqdm
import logging
import random
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

COLLECTION_NAME = "hybrid_search"
VECTOR_SIZE = 384  # for all-MiniLM-L6-v2
NUM_USERS = 10

class DataIngester:
    def __init__(self, host: str = "localhost", port: int = 6333):
        self.client = QdrantClient(
            host=host,
            port=port,
            prefer_grpc=True
        )
        # Set up Models
        self.client.set_model("sentence-transformers/all-MiniLM-L6-v2")
        self.client.set_sparse_model("prithivida/Splade_PP_en_v1")

    def create_collection(self) -> None:
        """Create collection with binary quantization and proper configuration."""
        try:
            if self.client.collection_exists(COLLECTION_NAME):
                logger.info(f"Recreating collection {COLLECTION_NAME}")
                self.client.delete_collection(COLLECTION_NAME)

            # Configuring binary quantization
            quantization_config = BinaryQuantization(
                binary=BinaryQuantizationConfig(
                    always_ram=True
                )
            )

            # Creating collection with both vector types
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=self.client.get_fastembed_vector_params(
                    quantization_config=quantization_config
                ),
                sparse_vectors_config=self.client.get_fastembed_sparse_vector_params(),
                shard_number=3,
                replication_factor=2,
                write_consistency_factor=2
            )

            # Creating payload index for user_id using correct schema type
            self.client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="user_id",
                field_schema=PayloadSchemaType.INTEGER  # Changed from "integer" to PayloadSchemaType.INTEGER
            )

            logger.info("Collection created successfully")

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise

    def ingest_data(self, documents: List[str], batch_size: int = 256) -> None:
        """Ingest documents with user_ids."""
        try:
            total_docs = len(documents)
            
            for i in tqdm(range(0, total_docs, batch_size), desc="Ingesting documents"):
                batch_docs = documents[i:i + batch_size]
                
                # Generating random user_ids for this batch
                batch_user_ids = [random.randint(1, NUM_USERS) for _ in range(len(batch_docs))]
                
                # Creating metadata for each document
                metadata = [
                    {"user_id": user_id, "text": doc}
                    for user_id, doc in zip(batch_user_ids, batch_docs)
                ]

                
                # Uploading batch
                self.client.add(
                    collection_name=COLLECTION_NAME,
                    documents=batch_docs,
                    metadata=metadata,
                    ids=range(i, i + len(batch_docs))
                )

            logger.info(f"Successfully ingested {total_docs} documents")

        except Exception as e:
            logger.error(f"Failed to ingest data: {e}")
            raise

def main():
    # Initializing ingester
    ingester = DataIngester()
    
    # Creating the collection
    ingester.create_collection()
    
    # Using HuggingFace datasets:
    from datasets import load_dataset
    
    dataset = load_dataset("wikipedia", "20220301.en", split="train")
    documents = [text for text in dataset["text"][:1000000]]  # Get first 1M documents
    
    # Ingesting data
    ingester.ingest_data(documents)

if __name__ == "__main__":
    main()
