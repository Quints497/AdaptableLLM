from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents.base import Document
from FlagEmbedding import FlagReranker


embeddings = HuggingFaceEmbeddings(model_name='llmrails/ember-v1', model_kwargs={'device': 'mps'})
vectorstore = Chroma(collection_name="testing", embedding_function=embeddings)
text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
document_loader = TextLoader("AdaptableLLM/src/shakespeare.txt").load()
split_documents = text_splitter.split_documents(document_loader)
vectorstore.add_documents(split_documents)

def format_k_documents(query_results: list[Document]) -> list:
    formatted_documents = f"\n{'-' * 100}\n".join(
        [f"Document {i}:\n\n" + docs.page_content for i, docs in enumerate(query_results)]
    )
    print(formatted_documents)
    return formatted_documents

def format_y_documents(query_results: list[Document]) -> list:
    formatted_documents = f"\n{'-' * 100}\n".join(
        [f"Document {i}: {pair['score']}\n\n" + pair['document'] for i, pair in enumerate(query_results)]
    )
    print(formatted_documents)
    return formatted_documents

def query_vectorstore(query: str, k: int) -> list[Document]:
    query_vector = embeddings.embed_query(query)
    results = vectorstore.similarity_search_by_vector(query_vector, k)
    return results

def rerank(query: str, query_results: list[Document], j: int) -> list[dict]:
    reranker = FlagReranker('BAAI/bge-reranker-large')
    scores = []
    for i, document in enumerate(query_results):
        score = reranker.compute_score([query, document.page_content])
        scores.append({
            'original_index': i, 
            'document': document.page_content, 
            'score': score
            })

    sorted_scores = sorted(scores, key=lambda x: x['score'], reverse=True)
    for i, pair in enumerate(sorted_scores):
        pair['sorted_index'] = i

    return sorted_scores

def show_changes(rerank_results: list[dict]):
    print(f"Original order \t\t\t   Sorted order \t\t    Original score \t\t\t\t Sorted score")
    for i, item in enumerate(rerank_results):
        print(f"\t{rerank_results[i]['original_index']} \t\t->\t\t {item['sorted_index']} \t\t---\t\t ({rerank_results[item['original_index']]['score']:.2f}) \t\t->\t\t ({item['score']:.2f})")

if __name__ == "__main__":
    query = "As an unperfect actor on the stage"
    k_documents = query_vectorstore(query, 15)
    format_k_documents(k_documents)
    print()
    print()
    print()
    print("#################\tReranking\t#################")
    print()
    print()
    print()
    y_reranked_documents = rerank(query, k_documents, 10)
    format_y_documents(y_reranked_documents)
    show_changes(y_reranked_documents)
