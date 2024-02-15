from assistant import Assistant
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from FlagEmbedding import FlagReranker

class RagAssistant:
    """
    Represents a RAG (Retrieval-Augmented Generation) model.

    Args:
        assistant (Assistant): The assistant used for generating responses.
        collection_name (str): The name of the collection in the vectorstore.
        directory (str): The directory where the vectorstore is stored.
        chunk_size (int): The size of each chunk when splitting the document.
        chunk_overlap (int): The overlap between chunks when splitting the document.
    """

    def __init__(self, 
                 assistant: Assistant = None, 
                 collection_name: str = "collection_name", 
                 directory: str = "directory", 
                 chunk_size: int = 1024, 
                 chunk_overlap: int = 0) -> None:
        """
        Initialize the RAG class.

        Args:
            assistant (Assistant): The assistant object.
            collection_name (str): The name of the collection.
            directory (str): The directory where the collection is stored.
            chunk_size (int): The size of each text chunk.
            chunk_overlap (int): The overlap between consecutive text chunks.
        """
        self.assistant = assistant
        self.embeddings = HuggingFaceEmbeddings(model_name='llmrails/ember-v1', model_kwargs={'device': 'mps'})
        self.vectorstore = Chroma(collection_name, self.embeddings, directory)
        self.text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    def add_document(self, document: str):
        """
        Add a document to the vectorstore.

        Parameters:
        document (str): The path to the document file.

        Returns:
        None
        """
        documents = TextLoader(document).load()
        split_documents = self.text_splitter.split_documents(documents)
        self.vectorstore.add_documents(documents=split_documents)
        print(f"Added {len(split_documents)} documents to the vectorstore.")

    def format_documents(self, query_results) -> list:
        """
        Formats the documents from the query results.

        Args:
            query_results (list): A list of documents retrieved from a query.

        Returns:
            str: The formatted documents as a string.
        """
        formatted_documents = f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + docs.page_content for i, docs in enumerate(query_results)]
        )
        print(formatted_documents)
        return formatted_documents

    def query_vectorstore(self, query: str, k: int):
        """
        Queries the vectorstore for similar vectors based on the given query.

        Args:
            query (str): The query string.
            k (int): The number of similar vectors to retrieve.

        Returns:
            list: A list of similar vectors.
        """
        query_vector = self.embeddings.embed_query(query)
        results = self.vectorstore.similarity_search_by_vector(query_vector, k)
        return results

    def rerank(self, query: str, query_results):
        reranker = FlagReranker('BAAI/bge-reranker-large')
        scores = []
        for document in query_results:
            score = reranker.compute_score([query, document.page_content])
            scores.append(score)
        return scores

    def start_rag_chat(self):
            """
            Starts a chat session with the RAG assistant.
            
            The method prompts the user for input, generates a response using the RAG assistant,
            and prints the response to the console. The chat session continues until the user
            enters "stop" as the prompt.
            """
            while True:
                prompt = input("Prompt: ")
                if prompt.lower() == "stop":
                    break
                query_result = self.query_vectorstore(query=prompt, k=3)
                print(self.rerank(query=prompt, query_results=query_result))
                context = self.format_documents(query_results=query_result)
                print()   
                print('-' * 100)
                output = ""
                print("AI:", end="")  
                for response in self.assistant.generate_response_from_prompt(context=context, prompt=prompt):
                    output += response
                    if output.strip().lower() == "Fuck off.".strip().lower():
                        break
                    print(response, end="", flush=True)
                print()


