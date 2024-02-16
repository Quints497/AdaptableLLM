from assistant import Assistant
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from FlagEmbedding import FlagReranker

class RagAssistant:
    """
    The RagAssistant class integrates a Retrieval-Augmented Generation (RAG) model with text embedding and vector storage for efficient document retrieval and response generation. It leverages a pre-trained language model for generating responses augmented by retrieved documents related to the query. This approach enhances the relevance and accuracy of the generated content.

    Attributes:
        assistant (Assistant): An instance of the Assistant class for generating responses.
        collection_name (str): The identifier for the document collection within the vector store.
        directory (str): File system path to the directory where the vector store and embeddings are saved.
        chunk_size (int): Determines how the documents are segmented into smaller parts for embedding.
        chunk_overlap (int): Specifies the character overlap between adjacent document segments to maintain context continuity.
        embeddings (HuggingFaceEmbeddings): Embedding model used for converting text to vector representations.
        vectorstore (Chroma): A vector store instance for storing and retrieving document embeddings.
        text_splitter (CharacterTextSplitter): Utility for splitting documents into manageable chunks based on size and overlap criteria.
    """
    def __init__(self, 
                 assistant: Assistant = None, 
                 collection_name: str = "collection_name", 
                 directory: str = "directory", 
                 chunk_size: int = 1024, 
                 chunk_overlap: int = 0) -> None:
        """
        Initializes a RagAssistant instance with specified configurations for document handling and response generation.

        Args:
            assistant (Assistant): The assistant object for generating textual responses.
            collection_name (str): Name of the collection within the vector store for document retrieval.
            directory (str): Path to the directory for storing vector store data.
            chunk_size (int): The size of each text chunk for document segmentation.
            chunk_overlap (int): Overlap size between consecutive text chunks for maintaining context.
        """
        self.assistant = assistant
        self.embeddings = HuggingFaceEmbeddings(model_name='llmrails/ember-v1', model_kwargs={'device': 'mps'})
        self.vectorstore = Chroma(collection_name, self.embeddings, directory)
        self.text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    def add_file(self, file: str):
        """
        Incorporates a new file into the vector store by segmenting it into chunks, embedding these chunks, and storing the resulting vectors.

        Args:
            file (str): The path to the file to be added.

        Returns:
            None, but prints the number of documents added to the vector store.
        """
        documents = TextLoader(file).load()
        split_documents = self.text_splitter.split_documents(documents)
        self.vectorstore.add_documents(documents=split_documents)
        print(f"Added {len(split_documents)} documents to the vectorstore.")

    def format_documents(self, query_results) -> list:
        """
        Formats retrieved documents from query results for display or further processing.

        Args:
            query_results (list): A list of documents retrieved from the vector store.

        Returns:
            str: A string representation of the formatted documents.
        """
        formatted_documents = f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + item['document'] for i, item in enumerate(query_results)]
        )
        print(formatted_documents)
        return formatted_documents

    def query_vectorstore(self, query: str, k: int):
        """
        Executes a similarity search in the vector store using the query's embedded representation to retrieve related documents.

        Args:
            query (str): The text query to search for.
            k (int): The number of similar documents to retrieve.

        Returns:
            list: A list of documents that are similar to the query.
        """
        query_vector = self.embeddings.embed_query(query)
        results = self.vectorstore.similarity_search_by_vector(query_vector, k)
        return results

    def rerank(self, query: str, query_results, j: int):
        """
        Re-ranks the initially retrieved documents based on a more sophisticated relevance assessment, potentially using additional models or criteria.

        Args:
            query (str): The original query text.
            query_results (list): The initial list of retrieved documents.
            j (int): The number of top documents to select after re-ranking.

        Returns:
            list: A list of re-ranked documents based on their relevance to the query.
        """
        reranker = FlagReranker('BAAI/bge-reranker-large')
        rerank_results = []
        for i, document in enumerate(query_results):
            result = reranker.compute_score([query, document.page_content])
            rerank_results.append({
                'original_index': i, 
                'document': document.page_content, 
                'score': result
                })

        reranked_results = sorted(rerank_results, key=lambda x: x['score'], reverse=True)
        for i, pair in enumerate(reranked_results):
            pair['sorted_index'] = i

        chosen_ranked_results = [item for item in reranked_results[:j] if item['score'] > 0]
        return chosen_ranked_results

    def show_changes(self, reranked_results: list[dict], j: int = 3):
        """
        Displays the effect of re-ranking on the order and selection of documents.

        Args:
            reranked_results (list[dict]): The results after re-ranking, including original indices, sorted indices, and scores.
            j (int): The number of documents to display.

        Prints:
            A comparison between the original and new rankings of documents, along with their scores.
        """
        print(f"Original order \t\t\t   Sorted order \t\t\t  Score")
        for item in reranked_results:
            original_index = item['original_index']
            sorted_index = item['sorted_index']
            sorted_score = item['score']
            print(f"\t{original_index} \t\t->\t\t {sorted_index} \t\t---\t\t  {sorted_score:.2f} \t (Chosen)")


    def start_rag_chat(self):
        """
        Initiates an interactive chat session that uses RAG for generating responses. The user inputs queries, and the system provides answers based on both retrieved documents and generative capabilities of the assistant.

        The session continues until the user inputs "stop".
        """
        while True:
            prompt = input("Prompt: ")
            if prompt.lower() == "stop":
                break
            query_result = self.query_vectorstore(query=prompt, k=10)
            reranked_results = self.rerank(query=prompt, query_results=query_result, j=3)
            context = self.format_documents(query_results=reranked_results)
            if context == "":
                print("No documents found.")
                continue
            self.show_changes(reranked_results)
            print()   
            print('-' * 100)
            output = ""
            print("AI: ", end="")  
            for response in self.assistant.generate_response_from_prompt(context=context, prompt=prompt):
                print(response, end="", flush=True)
                output += response
                if output.strip().lower() == "I don't know.".strip().lower():
                    break
            self.assistant.gather_statistics({"prompt": prompt, "output": output})
            print()