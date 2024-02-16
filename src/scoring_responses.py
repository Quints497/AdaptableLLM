# To implement a grading algorithm based on cosine similarity, we first need to prepare text representations
# of both the expected (ideal) responses and the actual responses, and then calculate the similarity between them.

from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

# Sample expected responses from a "superior machine" or ideal answers for comparison
# These are hypothetical examples based on the content and context of the original story
ideal_responses = [
    "The invention of the wheel revolutionized transportation and facilitated trade over 5,000 years ago, laying the groundwork for complex societies.",
    "The Industrial Revolution transformed societies from agrarian economies to industrial powerhouses through innovations like the steam engine, the power loom, and smelting iron, fueling unprecedented economic growth and social change.",
    "The invention of the computer marked the onset of the Information Age in the 20th century, leading to rapid developments in personal computing and the internet, revolutionizing communication, commerce, and information exchange.",
    "Artificial intelligence (AI) and quantum computing are at the forefront of the Digital Revolution, driving significant advancements in various sectors and blurring the lines between the physical and digital worlds.",
    "IoT is transforming our living environments by making smart homes, cities, and industries a reality. It connects devices and systems, allowing for smarter decision-making and more efficient resource management, improving energy efficiency, enhancing safety, and offering convenience.",
    "Blockchain technology offers the potential to revolutionize sectors by providing a decentralized and secure way to record transactions. It fosters trust in digital transactions, impacting various industries such as finance and supply chain management.",
    "According to the narrative, quantum computing represents the next frontier in computing technology, with the potential to process information at speeds unimaginable with current technology and solve complex problems that are currently intractable."
    "Advancements in AI and automation bring about ethical dilemmas, including concerns around privacy, security, and the impact of automation on employment. These technologies require careful navigation to balance technological progress with societal implications, ensuring equitable access and addressing the challenges posed by these transformative advancements."
]
filename = "rag-chat-history.json"
chat_history = json.load(open(filename))
# Extract actual responses from the chat history
actual_responses = [item['output'] for item in chat_history]

# Combine all responses to create the corpus
corpus = ideal_responses + actual_responses

embeddings = HuggingFaceEmbeddings(model_name='llmrails/ember-v1', model_kwargs={'device': 'mps'})
def generate_embeddings(texts):
    return np.array([embeddings.embed_query(text) for text in texts])

# Generate embeddings
ideal_embeddings = generate_embeddings(ideal_responses)
actual_embeddings = generate_embeddings(actual_responses)

# Calculate cosine similarity between ideal responses and actual responses
# Ideal responses are the first 7 documents in the matrix, and actual responses follow
scaled_scores = cosine_similarity(ideal_embeddings, actual_embeddings)

# Scale cosine similarity scores to 0-100 for grading
scaled_scores = np.around(scaled_scores.diagonal() * 100, 2)

# Add similarity scores as advanced grades to the chat history items
for item, score in zip(chat_history, scaled_scores):
    item['cosine_similarity_score'] = score

with open(filename, "w") as file:
   json.dump(chat_history, file, indent=4)