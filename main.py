from adapterAi import YiAdapter
from assistantAi import Assistant
from dotenv import load_dotenv
import os


if __name__ == "__main__":
    load_dotenv('config.env')
    model_path = os.getenv("MODEL_PATH")

    adapter = YiAdapter(model_path=model_path, 
                        n_gpu_layers=-1, 
                        n_batch=512, 
                        n_ctx=2048, 
                        verbose=False, 
                        prompt_template="<|im_start|>system\nYou are a helpful assistant. Who will attempt to speak like a pirate.<|im_end|>\n<|im_start|>user\n{prompt}\n<|im_end|><|im_start|>assistant")
    
    assistant = Assistant(adapter=adapter,
                          max_tokens=2048,
                          temperature=0.5,
                          top_p=0.1,
                          stream=True,
                          stop=["</s>", "<|im_end|>"])
    assistant.chat()
