import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

class GovernanceVault:
    def __init__(self, db_path="./data/chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # Create or get the collection for regulations
        self.collection = self.client.get_or_create_collection(name="eu_regs")

    def index_regulations(self, csv_path):
        df = pd.read_csv(csv_path)
        
        for i, row in df.iterrows():
            # Create a rich text string for the AI to "read"
            text_content = f"Regulation: {row['regulation']}. Requirement: {row['requirement']}. Application: {row['application']}"
            
            # Generate the vector embedding
            embedding = self.model.encode(text_content).tolist()
            
            # Add to the database
            self.collection.add(
                ids=[row['id']],
                embeddings=[embedding],
                documents=[text_content],
                metadatas=[{"source": row['regulation']}]
            )
        print(f"Indexed {len(df)} regulatory snippets.")

    def query(self, user_query, n_results=1):
        query_embedding = self.model.encode(user_query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results['documents'][0] if results['documents'] else ["No matching regulation found."]