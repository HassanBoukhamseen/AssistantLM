from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

GPT_CLIENT = OpenAI(
    api_key=os.getenv('OPEN_AI_API_KEY'), 
    project=os.getenv('PROJ')
)

CHATGPT3 = "gpt-3.5-turbo"
CHATGPT4 = "gpt-4"
LLAMA2 = "meta/llama-2-70b-chat"
FALCON2 = "joehoover/falcon-40b-instruct:7d58d6bddc53c23fa451c403b2b5373b1e0fa094e4e0d1b98c3d02931aa07173"

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
INDEX_NAME = "pwc-assessment"

SYSTEM_PROMPT =  '''
    You will be given a prompt and context information to use in answering it. 
    The prompts will primarily concern governmental processes in the United Arab Emirates. 
    Use only the provided context information to formulate your response, without referencing 
    any background knowledge. If the given context is not relevant or informative, or if 
    the question is not related to the UAE government and its processes, instruct 
    the user to defer to a human expert on the subject.
'''
RESPONSE_EVAL_SYSTEM_PROMPT = '''
    You will be given prompts and how an LLM model responded to said Prompt. 
    The prompts mainly have to do with governmental processes in the United Arab Emirates. 
    I want you to evaluate the response very critically. Your response should in one single paragraph. 
    Do not use headers or bullet points, just evaluate the response in one paragraph 
'''

ENGINEER_SYSTEM_PROMPT = '''
    You will now receive a prompt asking about a governmental process in the UAE. 
    Along with this prompt, the 15 most relevant entries taken from data scraped from UAE governmental 
    websites are included. 

    Your task:
    1. Phrase the context information given for the best coherence.
    2. Do not use any background information or add anything to the context in any way, shape, or form.
    3. Your response will be directly fed into another LLM. Therefore, do not include anything but the two parts requested: Coherently phrase the prompt and context information.
    4. Do not remove any details from the relevant context.
    5. Make the response concise (about 200 words).
    6. Start with the context first, then say: "Based on this info," followed by the prompt.
    7. If the given context is not as relevant or informative, or if the prompt is not specifically related to a governmental question in the UAE, instruct the LLM model to tell the user to defer to a human expert on the subject.
'''
