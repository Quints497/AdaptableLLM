from src.adapter import Adapter
from src.assistant import Assistant
from rag_assistant import RagAssistant

from dotenv import load_dotenv
import os


if __name__ == "__main__":
    load_dotenv('config.env')
    yi_model_path = os.getenv("YI_MODEL_PATH")
    mixtral_model_path = os.getenv("MIXTRAL_MODEL_PATH")
    mistral_model_path = os.getenv("MISTRAL_MODEL_PATH")
    
    prompt_template = "<|im_start|>system\n{system_message}\n<|im_end|><|im_start|>user\nContext: {context}\n Prompt: {prompt}\n<|im_end|><|im_start|>assistant"
    

    mistral_adapter = Adapter(model_path=mistral_model_path, 
                        n_gpu_layers=-1, 
                        n_batch=512, 
                        n_ctx=4096, 
                        verbose=False, 
                        prompt_template=prompt_template)
    
    mistral_assistant = Assistant(adapter=mistral_adapter,
                          max_tokens=2048,
                          temperature=0,
                          top_p=0.1,
                          stream=True,
                          stop=["</s>", "<|im_end|>"])
    
    rag_assistant = RagAssistant(assistant=mistral_assistant,
                        collection_name="rag-collection", 
                        directory="directory", 
                        chunk_size=1024, 
                        chunk_overlap=0)
    
    rag_assistant.add_document("/Users/om/Documents/AdaptableLLM/J. K. Rowling - Harry Potter 1 - Sorcerer's Stone.txt")
    rag_assistant.start_rag_chat()

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