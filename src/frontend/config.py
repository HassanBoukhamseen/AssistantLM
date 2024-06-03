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

SYSTEM_PROMPT = "You will be given a prompt and context information to use in answering it. The prompts will mainly concern governmental processes of the United Arab Emirates. You will not use your background knowledge, and only reference the context given in answering prompts."
RESPONSE_EVAL_SYSTEM_PROMPT = "You will be given prompts and how an LLM model responded to said Prompt. The prompts mainly have to do with governmental processes in the United Arab Emirates. I want you to evaluate the response very critically. Your response should in one single paragraph. Do not use headers or bullet points, just evaluate the response in one paragraph"