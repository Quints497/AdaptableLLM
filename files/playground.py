from adapterAi import Adapter
from assistantAi import Assistant
from dotenv import load_dotenv
import os

if __name__ == "__main__":
    load_dotenv('config.env')
    yi_model_path = os.getenv("YI_MODEL_PATH")
    mixtral_model_path = os.getenv("MIXTRAL_MODEL_PATH")
    
    prompt_template = "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|><|im_start|>user\n{prompt}\n<|im_end|><|im_start|>assistant"
    

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

    mixtral_adapter = Adapter(model_path=mixtral_model_path, 
                        n_gpu_layers=-1, 
                        n_batch=512, 
                        n_ctx=2048, 
                        verbose=False, 
                        prompt_template=prompt_template)
    
    mixtral_assistant = Assistant(adapter=mixtral_adapter,
                          max_tokens=2048,
                          temperature=0.5,
                          top_p=0.1,
                          stream=True,
                          stop=["</s>", "<|im_end|>"])
    
    mixtral_assistant.start_chat()
