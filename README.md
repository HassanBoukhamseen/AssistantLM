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
