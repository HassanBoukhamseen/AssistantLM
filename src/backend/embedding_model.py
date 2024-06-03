from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

model = SentenceTransformer('all-MiniLM-L6-v2')
text_embeddings = {}
lines = []
with open('collected_text.txt', 'r', encoding='utf-8') as file:
    for i, line in enumerate(file):
        progress = ((i + 1) / 82659) * 100
        print(f"\rProgress: {progress:.2f}%", end='')
        if line not in set(lines):
            lines.append(line)
            line = line.strip()
            if line:
                vector = model.encode(line)
                text_embeddings[i] = {"text": line, "vector": vector}

with open('text_embeddings.pkl', 'wb') as f:
    pickle.dump(text_embeddings, f)

print('\nText embeddings have been saved to text_embeddings.pkl')
