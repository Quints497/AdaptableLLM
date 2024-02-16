from datetime import datetime
import logging
from adapter import Adapter


class Assistant:
    """
    Implements an AI assistant for chat interactions, utilizing an LLM adapter for response generation.
    Maintains a chat history and logs interactions for analysis.

    Attributes:
        adapter (LLMAdapter): Adapter for generating responses.
        parameters (dict): Configuration for response generation including max_tokens, temperature, top_p, stream, and stop conditions.
        history (list): Records of chat interactions.
        logger (logging.Logger): Logger for interaction logging.
    """

    def __init__(self, 
                 adapter: Adapter = None, 
                 max_tokens: int = 1024, 
                 temperature: float = 0.5,
                 top_k: int = 10,
                 top_p: float = 0.1, 
                 stream: bool = True, 
                 stop: list = ["<|im_end|>"]) -> None:
        """
        Initializes the Assistant with a response generation adapter and configuration parameters.

        Args:
            adapter (LLMAdapter): Adapter for generating responses.
            max_tokens (int): Maximum number of tokens per response.
            temperature (float): Temperature for response generation randomness.
            top_p (float): Top-p for response generation diversity.
            stream (bool): Flag to stream output or return as a single string.
            stop (list): Conditions to stop response generation.
        """
        self.adapter = adapter
        self.parameters = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "stream": stream,
            "stop": stop
        }
        self.system_message = "Use the provided context to answer questions. If the answer cannot be found in the context, write 'I don't know.'"
        self.history = []
        self.__name__ = "Assistant"
        self.init_logging()

    def init_logging(self):
        """
        Sets up logging configuration with a file handler and specific formatter.
        """
        self.logger = logging.getLogger("Assistant-Logger")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(f"{__name__}-{self.__name__}.log")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Check to prevent adding duplicate handlers
        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)

    def update_system_message(self, message: str):
        # Update the system message for the assistant
        self.system_message = message

    def gather_statistics(self, messages: dict):
        """
        Logs user and assistant messages to the logger and updates the chat history.

        Args:
            messages (dict): Dictionary containing 'prompt' (user input) and 'output' (assistant's response).
        """
        self.logger.info(f"User: {messages['prompt']}")
        self.logger.info(f"Assistant: {messages['output']}")
        self.history.append(messages)

    def generate_response_from_prompt(self, context: str, prompt: str):
        """
        Generates a response for a given user prompt using the adapter.

        Args:
            prompt (str): The user's prompt.

        Returns:
            list: Generated responses.
        """
        formatted_prompt = self.adapter.prompt_format(system_message=self.system_message, context=context, prompt=prompt)
        output = self.adapter.invoke(prompt=formatted_prompt, **self.parameters)
        return self.adapter.parse_response(output=output)

    def handle_input(self, prompt: str):
        """
        Processes user input by generating a response, printing it, and logging the interaction.

        Args:
            prompt (str): The user's input.
        """
        output = ""
        for response in self.generate_response_from_prompt(prompt=prompt):
            print(response, end="", flush=True)
            output += response
        print()
        self.gather_statistics({"prompt": prompt, "output": output})

    def timestamp(self):
        """
        Returns the current timestamp in 'HH:MM:SS' format.

        Returns:
            str: The current timestamp.
        """
        return datetime.now().strftime('%H:%M:%S')

    def start_chat(self):
        """
        Initiates the chat loop, processing user input and generating responses until an exit command is issued.
        """
        while True:
            prompt = input(f"{self.timestamp()} - User: ")
            if prompt.lower() in ["exit", "goodbye"]:
                print("Assistant: Goodbye!")
                self.logger.info("Chat ended.")
                break

            print(f"{self.timestamp()} - Assistant: ", end="")
            self.handle_input(prompt=prompt)
