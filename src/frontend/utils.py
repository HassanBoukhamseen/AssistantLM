from pinecone import Pinecone
import time
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm
import replicate
import umap
import numpy as np
import spacy
from config import (
    GPT_CLIENT,
    CHATGPT3,
    CHATGPT4,
    LLAMA2,
    FALCON2,
    PINECONE_API_KEY,
    INDEX_NAME,
    SYSTEM_PROMPT,
    RESPONSE_EVAL_SYSTEM_PROMPT
)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
lemmatizer = spacy.load("en_core_web_sm")
encoder = SentenceTransformer('all-MiniLM-L6-v2')

lorum_ipsum = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book."
prompt = ""

def engineer_prompt(user_prompt, context):
    optimizer_prompt = f'''You will now recieve a prompt asking about a 
    governmental process in the UAE. Along this prompt, the 15 most 
    relevant entries taken from data scraped from UAE governmental 
    websites are included. Phrase the context information given for the best coherence.
    The prompt is:\n {user_prompt} \n and the scraped 
    context info is:\n {context}. 
    You are also NOT allowed to use background info or add into the contex in any way,
    shape, or form. Your resposne will be directly fed into another LLM. Therefore,
    I dont want your reply to include anything but the two parts I asked: 
    Coherently phrase prompt and context info. Do not remove any details from relevant context. Make your prompt concise (about 200 words). Start with the context first, then say: based on this info, then ask the prompt'''
    response = GPT_CLIENT.chat.completions.create(
        model=CHATGPT4,
        messages=[
            {"role": "user", "content": optimizer_prompt}
        ],
        stream=False
    )
    prompt_context = ''.join([choice.message.content for choice in response.choices])
    return prompt_context

def lemmatize_prompt(prompt):
    lemmatized_prompt = lemmatizer(prompt)
    lemmatized_prompt = ' '.join([token.lemma_ for token in lemmatized_prompt])
    return lemmatized_prompt

def evaluate_response(prompt, response):
    entire_promt = f"Here is the prompt: \n{prompt}\nAnd Here is the response:\n{response}"
    response = GPT_CLIENT.chat.completions.create(
        model=CHATGPT4,
        messages=[
            {"role": "system", "content": RESPONSE_EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": entire_promt}
        ],
        stream=False,
        temperature=0.2,
        seed=5
    )
    response = ''.join([choice.message.content for choice in response.choices])
    return response

def retrieve_context(prompt, top_k=3):
    query = encoder.encode(prompt)
    query_results = index.query(
        namespace="collected_text",
        vector=query.tolist(),
        top_k=top_k,
        include_metadata=True,
        include_values=True
    )
    context = ""
    db_output = {
        "scores": [],
        "context": [],
        "values": [],
    }
    with open("collected_text.txt", "r") as file:
        lines = file.readlines()
        for result in query_results['matches']:
            line_idx = int(result["metadata"]["line"])
            context += lines[line_idx] + " \n"
            db_output["scores"].append(result["score"])
            db_output["context"].append(lines[line_idx])
            db_output["values"].append(result['values'])
    engineered_prompt = engineer_prompt(prompt, context)
    lemmatized_prompt = lemmatize_prompt(engineered_prompt)
    return db_output, context, engineered_prompt, lemmatized_prompt

def get_embeddings(prompt, top_k=3):
    query = encoder.encode(prompt)
    query_results = index.query(
        namespace="collected_text",
        vector=query.tolist(),
        top_k=top_k,
        include_metadata=False,
        include_values=True
    )
    vector_results = []
    for result in query_results['matches']:
            vector_results.append(result['values'])
    vector_results = np.array(vector_results)
    data = np.concatenate((vector_results, query.reshape(1, -1)), axis=0)
    reducer = umap.UMAP(n_neighbors=top_k, min_dist=0.01, n_components=3, random_state=42)
    embedding = reducer.fit_transform(data)
    return embedding

def get_cgpt3_response(prompt_context):
    response = GPT_CLIENT.chat.completions.create(
        model=CHATGPT3,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_context}
        ],
        stream=False,
        temperature=0.2,
        seed=5
    )
    response = ''.join([choice.message.content for choice in response.choices])
    return response

def get_cgpt4_response(prompt_context):
    response = GPT_CLIENT.chat.completions.create(
        model=CHATGPT4,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_context}
        ],
        stream=False,
    )
    response = ''.join([choice.message.content for choice in response.choices])
    return response

def get_llama2_response(prompt_context):
    response = replicate.run(
        LLAMA2,
        input={
            "top_p": 1,
            "prompt": prompt_context,
            "temperature": 0.1,
            "system_prompt": SYSTEM_PROMPT,
            "max_new_tokens": 500
        }
    )
    response = ''.join(response)
    return response

def get_falcon2_response(prompt_context):
    response = replicate.run(
        FALCON2,
        input={
            "seed": 10,
            "top_p": 1,
            "prompt": "Only use the information provided in this prompt to answer." + "\n " + prompt_context,
            "max_length": 750,
            "temperature": 0.75,
            "length_penalty": 1
        }
    )
    response = ''.join([item for item in response])
    return response

def compute_score(prompt_embedding, response):
    response_embedding = encoder.encode(response)
    cos_sim = np.dot(prompt_embedding, response_embedding)/(norm(prompt_embedding)*norm(response_embedding))
    return float(cos_sim)
