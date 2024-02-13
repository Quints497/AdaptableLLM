from abstractAi import LLMAdapter
from llama_cpp import Llama
import logging

class YiAdapter(LLMAdapter):
    def __init__(self, model_path, n_gpu_layers, n_batch, n_ctx, verbose, prompt_template):
        """
        Initializes an instance of the YiAdapter class.

        Args:
            model_path (str): The path to the model.
            n_gpu_layers (int): The number of GPU layers.
            n_batch (int): The batch size.
            n_ctx (int): The context size.
            verbose (bool): Whether to enable verbose logging.
            prompt_template (str): The template for generating prompts.

        Returns:
            None
        """
        self.llm = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, n_batch=n_batch, n_ctx=n_ctx, verbose=verbose)
        self.prompt_template = prompt_template
        self.generating = False
        self.__name__ = "YiAdapter"
        self.init_logging()

    def init_logging(self):
        """
        Initializes the logging configuration.

        Args:
            None

        Returns:
            None
        """
        self.logger = logging.getLogger("YiLogger")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(f"{__name__}-{self.__name__}.log")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)

    def invoke(self, prompt: str, **parameters):
        """
        Invokes the adapter to generate a response.

        Args:
            prompt (str): The input prompt.
            **parameters: Additional parameters.

        Returns:
            str: The generated response.
        """
        self.generating = True
        self.logger.info(f"Generating: {self.generating}")
        output = self.llm(prompt, **parameters)
        return output
        
    def prompt_format(self, prompt: str) -> str:
        """
        Formats the prompt using the template.

        Args:
            prompt (str): The input prompt.

        Returns:
            str: The formatted prompt.
        """
        return self.prompt_template.format(prompt=prompt)

    def parse_response(self, output):
        """
        Parses and returns the response.

        Args:
            output: The output generated by the adapter.

        Yields:
            str: The parsed response.
        """
        for idx, chunk in enumerate(output):
            if idx == 0:
                continue
            yield chunk['choices'][0]['text']
        self.generating = False
        self.logger.info(f"Generating: {self.generating}")