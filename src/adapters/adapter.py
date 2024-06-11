from collections.abc import Generator, Iterator

from llama_cpp import CreateCompletionStreamResponse, Llama


class Adapter:
    """
    Adapter class that interfaces with the Llama model for generating responses based on provided prompts.

    Attributes:
        llm (Llama): Instance of the Llama model for response generation.
        prompt_template (str): Template string for formatting prompts.
        generating (bool): Flag indicating if response generation is in progress.
        logger (logging.Logger): Logger for recording adapter operations.
    """

    def __init__(
        self,
        model_path: str = "",
        n_gpu_layers: int = 0,
        n_batch: int = 512,
        n_ctx: int = 4096,
        verbose: bool = False,
        prompt_template: str = "<|im_start|>system\n{system_message}\n<|im_end|><|im_start|>user\nQuery: {query}\n<|im_end|><|im_start|>assistant",
    ) -> None:
        """
        Initializes the Adapter with Llama model parameters and a prompt template.

        Args:
            model_path (str): File system path to the pretrained model.
            n_gpu_layers (int): Number of layers to run on the GPU.
            n_batch (int): Number of tokens to process in parallel.
            n_ctx (int): Maximum number of tokens the model can consider for a single prompt.
            verbose (bool): Enables detailed logging if True.
            prompt_template (str): Template for generating prompts, with placeholders for dynamic content.
        """
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ctx=n_ctx,
            verbose=verbose,
        )
        self.prompt_template = prompt_template

    def invoke(
        self, prompt: str, **parameters: dict[str, any]
    ) -> Iterator[CreateCompletionStreamResponse]:
        """
        Generates a response from the Llama model based on the input prompt and additional parameters.

        Args:
            prompt (str): Input prompt for the model.
            **parameters: Model parameters.

        Returns:
            Generator: The generated response from the model.
        """
        return self.llm(prompt, **parameters)

    def prompt_format(self, system_message: str, query: str) -> str:
        """
        Formats the input prompt using the predefined template.

        Args:
            prompt (str): The user's input prompt.

        Returns:
            str: Formatted prompt ready for model processing.
        """
        return self.prompt_template.format(system_message=system_message, query=query)

    def parse_response(
        self, output: Iterator[CreateCompletionStreamResponse]
    ) -> Generator[any, any, None]:
        """
        Parses and yields each chunk of the generated response.

        Args:
            output: The raw output generated by the Llama model.

        Yields:
            str: Each parsed chunk of the response.
        """
        for idx, chunk in enumerate(output):
            if idx == 0:  # Skip the first chunk if needed
                continue
            yield chunk["choices"][0]["text"]
