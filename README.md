# AssistantLM: Interactive Chat and Analysis System

AssistantLM is an AI-driven chat application designed to provide users with insightful and context-aware responses. Utilizing multiple advanced language models, this system retrieves relevant information from a vector database, refines user prompts, and generates optimized responses. The application features real-time streaming of responses, a comprehensive analysis page to review the performance of different models, and a 3D visualization of database vectors. Ideal for users seeking an interactive, informative, and efficient AI-assisted conversation experience.

## Website Design

1. **Home Page**: The main page establishes a mature product and a brand identity, welcoming users to the AssistantLM platform. 

<p align="center">
  <img src="https://github.com/HassanBoukhamseen/AssistantLM/assets/60622951/201db1ef-7404-48c7-a682-5185d49a2b8d" />
</p>

2. **Chat Page**: The chat page allows users to send messages and receive responses from four different LLM models. A machine learning model automatically determines the best response.

<p align="center">
  <img src="https://github.com/HassanBoukhamseen/AssistantLM/assets/60622951/e03b3551-a7f6-412c-9dcb-f4a62f73ac37" />
</p>

3. **Chat History Analysis Page**: Clicking on the "Full Analysis" button in the chat page takes the user to a page that highlights the inner workings of the models. 
  - **3.1 Prompt Evolution**: A user prompt is first augmented with vectors that match the encoded prompt from the database. An LLM processes this prompt along with the contextual information to craft a coherent response. Finally, a lemmatizer optimizes the output, reducing the number of tokens and ensuring a cost-effective operation.

<p align="center">
  <img src="https://github.com/HassanBoukhamseen/AssistantLM/assets/60622951/027e6500-4183-4e78-bc6a-416b14c863c5" />
</p>
  
  - **3.2 LLM Response Evaluation**: An LLM evaluates the responses of all the LLMs that attempt to answer the user’s prompt, providing detailed performance metrics for each model.

<p align="center">
  <img src="https://github.com/HassanBoukhamseen/AssistantLM/assets/60622951/6b794646-c3dc-45aa-87e3-1c1ee8e807fd" />
</p>
  
  - **3.3 Database Vector Selection and Visualization**: Vectors stored in the database and the prompt are mapped to a lower dimensional space (3D) using UMAP to visualize the embeddings. This visualization, along with a table showing retrieved textual information and cosine similarity scores, helps users understand the model’s capabilities and the relevance of the information in the database. 

<p align="center">
  <img src="https://github.com/HassanBoukhamseen/AssistantLM/assets/60622951/af336a16-9ed8-4594-ab2f-15bbe5fb641b" />
</p>

## Setup

### 1. Create a Python Environment

Create a new Python environment using your preferred method. For example, using `venv`:
```sh
python3 -m venv AssistantLM-env
cd AssistantLM-env
```
### 2. Clone the Repository
Clone the repository to your local machine:
```sh
git clone <Repository_URL>
```
### 3. Navigate to the Project Directory
Change directory to the project directory:
```sh
cd AssistantLM
```
### 4. Activate Environment
Activate the environment using the following command:
```sh
source bin/activate
```
### 5. Install Dependencies
Install the required Python packages:
```sh
pip install -r requirements.txt
pip install python-dotenv
python -m spacy download en_core_web_sm
```
### 6. Set Up Environment Variables
Create a .env file in the project directory and add your API keys and sensitive information:
```env
OPEN_AI_API_KEY=<Key>
PINECONE_API_KEY=<Key>
PROJ=<Proj>
```
### 7. Export Replicate API Token
Export your Replicate API token:
```sh
export REPLICATE_API_TOKEN=<API_Token>
```
### 8. Navigate to the Frontend Directory
Change directory to the frontend source directory:
```sh
cd src/frontend
```
### 9. Run the Application
Run the application using Python:
```sh
python3 main.py
```
## Tools

1. **Pinecone**: Pinecone is an excellent choice for vector databases due to its reliability and ease of implementation. It allows efficient and scalable storage and retrieval of high-dimensional vectors, making it perfect for machine learning and AI applications.

2. **Sentence Transformer**: The Sentence Transformer encoder model, which uses BERT under the hood, is preferred over other encoders due to its speed and effectiveness in generating high-quality embeddings. It captures semantic meaning effectively, enhancing the relevance of retrieved information.

3. **Selenium and BeautifulSoup for Web Scraping**: 
  - **Selenium**: Selenium is a powerful tool for web automation, allowing interaction with web pages and extraction of dynamic content.
  - **BeautifulSoup**: BeautifulSoup is used for parsing HTML and XML documents, providing Pythonic idioms for iterating, searching, and modifying the parse tree.
