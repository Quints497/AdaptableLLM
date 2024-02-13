from abstractAi import LLMAdapter
from datetime import datetime
import logging


class Assistant:
    """
    The Assistant class represents an AI assistant that interacts with users through a chat interface.
    It uses an adapter to generate responses based on user prompts and gathers statistics from the chat.

    Args:
        adapter: The adapter used to generate responses.
        max_tokens: The maximum number of tokens to generate for each response.
        temperature: The temperature value for controlling the randomness of the generated responses.
        top_p: The top-p value for controlling the diversity of the generated responses.
        stream: A boolean indicating whether to stream the output or return it as a single string.
        stop: A string indicating the stopping condition for generating responses.

    Attributes:
        adapter: The adapter used to generate responses.
        parameters: A dictionary containing the parameters for response generation.
        history: A list of dictionaries representing the chat history.
        logger: The logger object for logging chat messages.

    Methods:
        init_logging: Initializes the logging configuration.
        gather_statistics: Gathers statistics from the chat.
        generate_response_from_prompt: Generates a response from a user prompt.
        chat: Starts the chat interaction loop.
    """

    def __init__(self, adapter: LLMAdapter, max_tokens, temperature, top_p, stream, stop) -> None:
        self.adapter = adapter
        self.parameters = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "stop": stop
        }
        self.history = []
        self.__name__ = "LLM-Assistant"
        self.init_logging()

    def init_logging(self):
        """
        Initializes the logging configuration for the Assistant.
        """
        self.logger = logging.getLogger("LLM-Assistant-Logger")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(f"{__name__}-{self.__name__}.log")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)

    def gather_statistics(self, messages: dict):
        """
        Gathers statistics from the chat.

        Args:
            messages: A dictionary containing the user prompt and assistant output.
        """
        self.logger.info(f"User: {messages['prompt']}")
        self.logger.info(f"Assistant: {messages['output']}")
        self.history.append(
            {
                "User": messages['prompt'], 
                "Assistant": messages['output']
            }
        )

    def generate_response_from_prompt(self, prompt: str):
        """
        Generates a response from a user prompt.

        Args:
            prompt: The user prompt.

        Returns:
            A list of generated responses.
        """
        formatted_prompt = self.adapter.prompt_format(prompt)
        output = self.adapter.invoke(formatted_prompt, **self.parameters)
        return self.adapter.parse_response(output)
    
    def handle_user_input(self, prompt: str):
        """
        Handles the user input by generating a response and gathering statistics.

        Args:
            prompt: The user prompt.
        """
        output = ""
        for response in self.generate_response_from_prompt(prompt=prompt):
            print(response, end="", flush=True)
            output += response
        print()
        self.gather_statistics({"prompt": prompt, "output": output})

    def start_chat(self):
        """
        Starts the chat interaction loop.
        """
        while True:

            prompt = input(f"{datetime.now().strftime('%H:%M:%S')} - User: ")

            if prompt.lower() in ["exit", "goodbye"]:
                print("Assistant: Goodbye!")
                self.logger.info("Chat ended.")
                break
            
            print(f"{datetime.now().strftime('%H:%M:%S')} - Assistant: ", end="")
            self.handle_user_input(prompt=prompt)

