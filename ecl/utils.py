import subprocess
import json
import yaml
import time
import logging
from easydict import EasyDict
import openai
from openai import OpenAI
import numpy as np
import os
from abc import ABC, abstractmethod
import tiktoken
from typing import Any, Dict
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential
)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if 'BASE_URL' in os.environ:
    BASE_URL = os.environ['BASE_URL']
else:
    BASE_URL = None

def getFilesFromType(sourceDir, filetype):
    files = []
    for root, directories, filenames in os.walk(sourceDir):
        for filename in filenames:
            if filename.endswith(filetype):
                files.append(os.path.join(root, filename))
    return files

def cmd(command: str):
    print(">> {}".format(command))
    text = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE).stdout
    return text

def get_easyDict_from_filepath(path: str):
    # print(path)
    if path.endswith('.json'):
        with open(path, 'r', encoding="utf-8") as file:
            config_map = json.load(file, strict=False)
            config_easydict = EasyDict(config_map)
            return config_easydict
    if path.endswith('.yaml'):
        file_data = open(path, 'r', encoding="utf-8").read()
        config_map = yaml.load(file_data, Loader=yaml.FullLoader)
        config_easydict = EasyDict(config_map)
        return config_easydict
    return None


def calc_max_token(messages, model):
    string = "\n".join([message["content"] for message in messages])
    encoding = tiktoken.encoding_for_model(model)
    num_prompt_tokens = len(encoding.encode(string))
    gap_between_send_receive = 50
    num_prompt_tokens += gap_between_send_receive

    # Context windows (roughly). Conservative defaults for unknown models.
    context_window_map = {
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-0613": 4096,
        "gpt-3.5-turbo-0125": 16384,
        "gpt-3.5-turbo-16k-0613": 16384,
        "gpt-4": 8192,
        "gpt-4-0613": 8192,
        "gpt-4-32k": 32768,
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
    }
    # Completion caps (output tokens). This is what `max_tokens` must obey.
    max_output_map = {
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-0613": 4096,
        "gpt-3.5-turbo-0125": 4096,
        "gpt-3.5-turbo-16k-0613": 4096,
        "gpt-4": 4096,
        "gpt-4-0613": 4096,
        "gpt-4-32k": 4096,
        "gpt-4o": 16384,
        "gpt-4o-mini": 16384,
    }

    context_window = context_window_map.get(model, 8192)
    max_output = max_output_map.get(model, 4096)
    available_by_context = max(0, context_window - num_prompt_tokens)
    safe = min(int(max_output), int(available_by_context))
    return max(1, safe)


class ModelBackend(ABC):
    r"""Base class for different model backends.
    May be OpenAI API, a local LLM, a stub for unit tests, etc."""

    @abstractmethod
    def run(self, *args, **kwargs) -> Dict[str, Any]:
        r"""Runs the query to the backend model.

        Raises:
            RuntimeError: if the return value from OpenAI API
            is not a dict that is expected.

        Returns:
            Dict[str, Any]: All backends must return a dict in OpenAI format.
        """
        pass

class OpenAIModel(ModelBackend):
    r"""OpenAI API in a unified ModelBackend interface."""

    def __init__(self, model_type, model_config_dict: Dict=None) -> None:
        super().__init__()
        self.model_type = model_type
        self.model_config_dict = model_config_dict
        if self.model_config_dict == None:
            self.model_config_dict = {"temperature": 0.2,
                                "top_p": 1.0,
                                "n": 1,
                                "stream": False,
                                "frequency_penalty": 0.0,
                                "presence_penalty": 0.0,
                                "logit_bias": {},
                                }
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    @retry(wait=wait_exponential(min=5, max=60), stop=stop_after_attempt(5))
    def run(self, messages) :
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set in the environment. "
                "Please export OPENAI_API_KEY before running."
            )
        if BASE_URL:
            client = openai.OpenAI(
                api_key=OPENAI_API_KEY,
                base_url=BASE_URL,
            )
        else:
            client = openai.OpenAI(
                api_key=OPENAI_API_KEY
            )
        current_retry = 0
        max_retry = 5

        string = "\n".join([message["content"] for message in messages])
        encoding = tiktoken.encoding_for_model(self.model_type)
        num_prompt_tokens = len(encoding.encode(string))
        gap_between_send_receive = 15 * len(messages)
        num_prompt_tokens += gap_between_send_receive

        # Context windows (roughly). Conservative defaults for unknown models.
        context_window_map = {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-0613": 4096,
            "gpt-3.5-turbo-0125": 16384,
            "gpt-3.5-turbo-16k-0613": 16384,
            "gpt-4": 8192,
            "gpt-4-0613": 8192,
            "gpt-4-32k": 32768,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
        }
        # Completion caps (output tokens). This is what `max_tokens` must obey.
        max_output_map = {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-0613": 4096,
            "gpt-3.5-turbo-0125": 4096,
            "gpt-3.5-turbo-16k-0613": 4096,
            "gpt-4": 4096,
            "gpt-4-0613": 4096,
            "gpt-4-32k": 4096,
            "gpt-4o": 16384,
            "gpt-4o-mini": 16384,
        }

        context_window = context_window_map.get(self.model_type, 8192)
        max_output = max_output_map.get(self.model_type, 4096)
        available_by_context = max(0, context_window - num_prompt_tokens)
        safe_max_tokens = min(int(max_output), int(available_by_context))
        safe_max_tokens = max(1, safe_max_tokens)

        response = client.chat.completions.create(messages = messages,
        model = self.model_type,
        max_tokens = safe_max_tokens,
        temperature = 0.2,
        top_p = 1.0,
        n = 1,
        stream = False,
        frequency_penalty = 0.0,
        presence_penalty = 0.0,
        logit_bias = {},
        ).model_dump()
        response_text = response['choices'][0]['message']['content']

        self.model_config_dict["max_tokens"] = safe_max_tokens
        log_and_print_online(
            "InstructionStar generation:\n**[OpenAI_Usage_Info Receive]**\nprompt_tokens: {}\ncompletion_tokens: {}\ntotal_tokens: {}\n".format(
                response["usage"]["prompt_tokens"], response["usage"]["completion_tokens"],
                response["usage"]["total_tokens"]))
        self.prompt_tokens += response["usage"]["prompt_tokens"]
        self.completion_tokens += response["usage"]["completion_tokens"]
        self.total_tokens += response["usage"]["total_tokens"]
        
        if not isinstance(response, Dict):
            raise RuntimeError("Unexpected return from OpenAI API")
        return response

    
def now():
    return time.strftime("%Y%m%d%H%M%S", time.localtime())

def log_and_print_online(content=None):
    if  content is not None:
        print(content)
        logging.info(content)
