from pprint import pprint
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
minilm = 'sentence-transformers/all-MiniLM-L6-v2'
rails = 'llmrails/ember-v1'
model_kwargs = {'device': 'mps'}
embeddings = [HuggingFaceEmbeddings(model_name=minilm, model_kwargs=model_kwargs), HuggingFaceEmbeddings(model_name=rails, model_kwargs=model_kwargs)]
    
text = 'Hello there'

for embedding in embeddings:
    print('\n' * 10)
    pprint(embedding)
    pprint(embedding.embed_query(text))
    pprint(len(embedding.embed_query(text)))

