# To implement a grading algorithm based on cosine similarity, we first need to prepare text representations
# of both the expected (ideal) responses and the actual responses, and then calculate the similarity between them.

import json

import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


def score_responses():
    ideal_file = "testing-rag.json"
    ideal_history = json.load(open(ideal_file))
    ideal_responses = [item["answer"] for item in ideal_history]

    chat_file = "rag-chat-history.json"
    chat_history = json.load(open(chat_file))
    # Extract actual responses from the chat history
    actual_responses = [item["output"] for item in chat_history]

    embeddings = HuggingFaceEmbeddings(
        model_name="llmrails/ember-v1", model_kwargs={"device": "mps"}
    )

    def generate_embeddings(texts):
        return np.array([embeddings.embed_query(text) for text in texts])

    # Generate embeddings
    ideal_embeddings = generate_embeddings(ideal_responses)
    actual_embeddings = generate_embeddings(actual_responses)

    # Calculate cosine similarity between ideal responses and actual responses
    scaled_scores = cosine_similarity(ideal_embeddings, actual_embeddings)

    # Scale cosine similarity scores to 0-100 for grading
    scaled_scores = np.around(scaled_scores.diagonal() * 100, 2)

    # Add similarity scores as advanced grades to the chat history items
    for item, response, score in zip(chat_history, ideal_responses, scaled_scores):
        item["ideal_response"] = response
        item["cosine_similarity_score"] = score

    with open("scored-rag-chat-history.json", "w") as file:
        json.dump(chat_history, file, indent=4)
