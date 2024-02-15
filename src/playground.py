from adapter import Adapter
from assistant import Assistant
from rag_assistant import RagAssistant

from dotenv import load_dotenv
import os



if __name__ == "__main__":
    load_dotenv('AdaptableLLM/config.env')
    yi_model_path = os.getenv("YI_MODEL_PATH")
    mixtral_model_path = os.getenv("MIXTRAL_MODEL_PATH")
    mistral_model_path = os.getenv("MISTRAL_MODEL_PATH")
    nous_model_path = os.getenv("NOUS_MODEL_PATH")
    # mistral_model_path = "mistral-7b-instruct-v0.2.Q6_K.gguf"

    nous_adapter = Adapter(model_path=nous_model_path, 
                        n_gpu_layers=-1, 
                        n_batch=512, 
                        n_ctx=4096, 
                        verbose=False,)
    
    nous_assistant = Assistant(adapter=nous_adapter,
                          max_tokens=2048,
                          temperature=0,
                          top_k=5,
                          top_p=0.1,
                          stream=True,
                          stop=["</s>", "<|im_end|>"])
    
    rag_assistant = RagAssistant(assistant=nous_assistant,
                        collection_name="rag-collection", 
                        directory="directory", 
                        chunk_size=1024, 
                        chunk_overlap=0)
    
    # rag_assistant.add_document("AdaptableLLM/src/shakespeare.txt")
    rag_assistant.start_rag_chat()



    # query = "When forty winters shall"
    # results = rag_assistant.query_vectorstore(query, 10)
    # rag_assistant.format_documents(results)
    # print(rag_assistant.rerank(query, results))


    # yi_adapter = Adapter(model_path=yi_model_path, 
    #                     n_gpu_layers=-1, 
    #                     n_batch=512, 
    #                     n_ctx=2048, 
    #                     verbose=False, 
    #                     prompt_template=yi_prompt_template)
    
    # yi_assistant = Assistant(adapter=yi_adapter,
    #                       max_tokens=2048,
    #                       temperature=0.5,
    #                       top_p=0.1,
    #                       stream=True,
    #                       stop=["</s>", "<|im_end|>"])
    
    # yi_assistant.start_chat()
    # mixtral_assistant.start_chat()

    # mixtral_adapter = Adapter(model_path=mixtral_model_path, 
    #                     n_gpu_layers=-1, 
    #                     n_batch=512, 
    #                     n_ctx=4096, 
    #                     verbose=False, 
    #                     prompt_template=prompt_template)
    
    # mixtral_assistant = Assistant(adapter=mixtral_adapter,
    #                       max_tokens=2048,
    #                       temperature=0,
    #                       top_p=0.1,
    #                       stream=True,
    #                       stop=["</s>", "<|im_end|>"])
    
    # rag_assistant = Rag(assistant=mixtral_assistant,
    #                     collection_name="rag-collection", 
    #                     directory="directory", 
    #                     chunk_size=2048, 
    #                     chunk_overlap=0)