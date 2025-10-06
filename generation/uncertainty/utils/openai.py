import os
import hashlib
from tenacity import retry, wait_random_exponential, retry_if_not_exception_type
from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()
CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


class KeyError(Exception):
    """OpenAIKey not provided in environment variable."""
    pass


@retry(retry=retry_if_not_exception_type(KeyError), wait=wait_random_exponential(min=1, max=10))
def predict(prompt, temperature=0.8, model='gpt-4'):
    """Predict with GPT models."""

    if not CLIENT.api_key:
        raise KeyError('Need to provide OpenAI API key in environment variable `OPENAI_API_KEY`.')

    if isinstance(prompt, str):
        messages = [
            {'role': 'user', 'content': prompt},
        ]
    else:
        messages = prompt

    if model == 'gpt-4':
        model = 'gpt-4-0613'
    elif model == 'gpt-4-turbo':
        model = 'gpt-4-turbo'
    elif model == 'gpt-3.5':
        model = 'gpt-3.5-turbo-1106'
    elif model == 'o1':
        model == 'o1-preview'
    elif model == 'gpt-4.1':
        model = 'gpt-4.1-2025-04-14'

    output = CLIENT.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=200,
        temperature=temperature,
    )
    response = output.choices[0].message.content
    return response


def md5hash(string):
    return int(hashlib.md5(string.encode('utf-8')).hexdigest(), 16)
