from flask import Flask, render_template, request, redirect, url_for, jsonify
import json
from time import time, sleep
from flask_socketio import SocketIO, emit
import threading
from utils import (
    retrieve_context,
    get_cgpt3_response,
    get_cgpt4_response,
    get_llama2_response,
    get_falcon2_response,
    evaluate_response,
    compute_score,
    get_embeddings,
    encoder
)

app = Flask(__name__)
app.config["SECRET_KEY"] = "soso"
socketio = SocketIO(app, async_mode="threading")

global_prompt = ""
cos_similarities = None

response_lock = threading.Lock()

def generate_response_thread(model, func, lemmatized_prompt, prompt_embedding, user_prompt, context, db_output, engineered_prompt, sid):
    t1 = time()
    response = func(engineered_prompt)
    score = compute_score(prompt_embedding, response)
    with response_lock:
        socketio.emit(
            "receive_message",
            {
                model: response,
                "score": score,
                "prompt": user_prompt,
                "context": context,
                "cos_similarities": db_output,
                "engineered_prompt": engineered_prompt,
                "lemmatized_prompt": lemmatized_prompt
            },
            to=sid
        )
        print(f"{model} response sent. Time: {time() - t1}")
        sleep(1)

def generate_responses(user_prompt, sid):
    t1 = time()
    db_output, context, engineered_prompt, lemmatized_prompt = retrieve_context(user_prompt, top_k=15)
    print(f"collected context in {time() - t1}")
    response_functions = [
        ("cgpt3", get_cgpt3_response),
        ("cgpt4", get_cgpt4_response),
        ("llama2", get_llama2_response),
        ("falcon2", get_falcon2_response)
    ]
    prompt_embedding = encoder.encode(engineered_prompt)
    threads = []
    print("PROMPT: ", engineered_prompt, "End PROMPT\n\n\n")
    for model, func in response_functions:
        thread = threading.Thread(target=generate_response_thread, args=(model, func, lemmatized_prompt, prompt_embedding, user_prompt, context, db_output, engineered_prompt, sid))
        thread.start()
        threads.append(thread)

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    with response_lock:
        socketio.emit("all_sent", to=sid)
        print(f"All responses sent. Time taken: {time() - t1}")

@socketio.on("send_message")
def handle_message(data):
    user_prompt = data["userMessage"]
    sid = request.sid
    print(f"Received message: {user_prompt}")
    thread = threading.Thread(target=generate_responses, args=(user_prompt, sid))
    thread.start()
    emit("progress_update", {"status": "processing"})

@socketio.on("get_embeddings")
def send_embeddings():
    global global_prompt, cos_similarities
    embedding = get_embeddings(global_prompt, top_k=15)
    embedding = embedding.tolist()
    embedding_json = json.dumps(embedding)
    print("Sending...")
    emit("receive_embeddings", embedding_json)
    cos_similarities = json.dumps(cos_similarities)
    emit("receive_scores", cos_similarities)

@app.route("/full_analysis", methods=["POST"])
def analysis():
    global global_prompt, cos_similarities
    chat_hist = json.loads(request.form["chatHist"])
    global_prompt = prompt = chat_hist["prompt"]
    context = chat_hist["context"]
    cos_similarities = chat_hist["cos_similarities"]
    engineered_prompt = chat_hist["engineered_prompt"]
    lemmatized_prompt = chat_hist["lemmatized_prompt"]
    cgpt3_eval = evaluate_response(prompt, chat_hist["cgpt3"])
    cgpt4_eval = evaluate_response(prompt, chat_hist["cgpt4"])
    llama2_eval = evaluate_response(prompt, chat_hist["llama2"])
    falcon2_eval = evaluate_response(prompt, chat_hist["falcon2"])
    return render_template(
        "analysis.html",
        prompt=prompt,
        context=context,
        engineered_prompt=engineered_prompt,
        lemmatized_prompt=lemmatized_prompt,
        cgpt3_response=cgpt3_eval,
        cgpt4_response=cgpt4_eval,
        llama2_response=llama2_eval,
        falcon2_response=falcon2_eval
    )

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    prompt = request.form["prompt"]
    return render_template("chat.html", prompt=prompt)

@app.route("/submit", methods=["POST"])
def submit_form():
    if request.method == "POST":
        prompt = request.form["prompt"]
        return redirect(url_for("chat"), code=307)
    return redirect(url_for("home"))

if __name__ == "__main__":
    socketio.run(app, debug=True, port=8000)
