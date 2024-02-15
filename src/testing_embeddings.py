from pprint import pprint
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model_name = 'llmrails/ember-v1'
model_kwargs = {'device': 'mps'}
embeddings = [HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)]
    
text = 'Hello there'

for embedding in embeddings:
    print()
    pprint(embedding.embed_query(text))
    pprint(len(embedding.embed_query(text)))

