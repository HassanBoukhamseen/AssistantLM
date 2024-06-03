from pinecone import Pinecone, ServerlessSpec
import time
import pickle

pc = Pinecone(api_key='fbef0f98-ec03-4ecb-a107-58ad82bcb424')
index_name = "pwc-assessment"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    ) 

index = pc.Index(index_name)
time.sleep(1)

with open('text_embeddings.pkl', 'rb') as f:
    text_embeddings = pickle.load(f)

for i, key in enumerate(text_embeddings.keys()):
    progress = ((i + 1) / len(text_embeddings)) * 100
    print(f"\r{i}) Progress: {progress:.2f}%", end='')
    vector = text_embeddings[key]['vector']
    text =  text_embeddings[key]['text']
    index.upsert(
        vectors=[{"id": str(i), "values": vector.tolist(), "metadata": {"line": key}}],
        namespace="collected_text"
    )

print(" ")
print(index.describe_index_stats())

