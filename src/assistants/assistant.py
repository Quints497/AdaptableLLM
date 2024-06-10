from adaptable.AdaptableLLM.src.adapters.adapter import Adapter
from collections.abc import Generator
import json
import gradio as gr

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
                 stop: list = ["<|im_end|>", "<|im_start|>"]) -> None:
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
        self.system_message = "You are Hermes 2, a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia."
        self.history = []


    def gather_statistics(self, messages: dict) -> None:
        """
        Updates the chat history.

        Args:
            messages (dict): Dictionary containing 'query' (user input) and 'output' (assistant's response).
        """
        self.history.append(messages)

    def export_statistics(self, filename: str) -> None:
        """
        Exports the chat history to a file.

        Args:
            filename (str): The name of the file to export the chat history to.
        """
        with open(filename, "w") as file:
            json.dump(self.history, file, indent=4)


    def generate_response_from_query(self, query: str) -> Generator[any, any, None]:
        """
        Generates a response for a given user query using the adapter.

        Args:
            query (str): The user's query.

        Returns:
            list: Generated responses.
        """
        formatted_prompt = self.adapter.prompt_format(system_message=self.system_message, query=query)
        output = self.adapter.invoke(prompt=formatted_prompt, **self.parameters)
        return self.adapter.parse_response(output=output)

    def handle_input(self, query: str, history):
        """
        Processes user input by generating a response, printing it, and logging the interaction.

        Args:
            prompt (str): The user's input.
        """
        output = ""
        for response in self.generate_response_from_query(query=query):
            print(response, end="", flush=True)
            output += response
            yield output
        print()
        self.gather_statistics({"query": query, "output": output})

    def start_chat(self) -> None:
        """
        Initiates the chat loop, processing user input and generating responses until an exit command is issued.
        """
        while True:
            prompt = input("User: ")
            if prompt.lower() in ["exit", "goodbye"]:
                print("Assistant: Goodbye!")
                break

            print("Assistant: ", end="")
            self.handle_input(query=prompt)

    def gradio_chat(self) -> None:
        gr.ChatInterface(self.handle_input).launch(server_name="0.0.0.0", server_port=8080)