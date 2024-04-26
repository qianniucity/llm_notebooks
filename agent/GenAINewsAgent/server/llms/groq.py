from groq import Groq, AsyncGroq
import traceback
from typing import List, Dict, Union
from llms.base import BaseLLM
# from llms.fuzzy_anthropic_ctx import manageContext
from llms.ctx import ContextManagement
from groq import RateLimitError
import backoff

manageContext = ContextManagement()


class GroqLLM(BaseLLM):

    def __init__(self, api_key: Union[str, None] = None):
        super().__init__(api_key)
        self.client = AsyncGroq(api_key=api_key)

    @backoff.on_exception(backoff.expo, RateLimitError, max_tries=3)
    async def __call__(self, model: str, messages: List[Dict], **kwargs):
        try:
            if "system" in kwargs:
                messages = [{
                    "role": "system",
                    "content": kwargs.get("system")
                }] + messages
                del kwargs["system"]
            if "ctx_length" in kwargs:
                del kwargs["ctx_length"]
            messages = manageContext(messages, kwargs.get("ctx_length", 7_000))
            output = await self.client.chat.completions.create(
                messages=messages, model=model, **kwargs)
            return output.choices[0].message.content
        except RateLimitError:
            raise RateLimitError
        except Exception as err:
            print(f"ERROR: {str(err)}")
            print(f"{traceback.format_exc()}")
            return ""


class GroqLLMStream(BaseLLM):

    def __init__(self, api_key: Union[str, None] = None):
        super().__init__(api_key)
        self.client = AsyncGroq(api_key=api_key)

    async def __call__(self, model: str, messages: List[Dict], **kwargs):
        if "system" in kwargs:
            # print(f"System in Args")
            messages = [{
                "role": "system",
                "content": kwargs.get("system")
            }] + messages
            del kwargs["system"]
        # print(f"KWARGS KEYS: {kwargs.keys()}")
        messages = manageContext(messages, kwargs.get("ctx_length", 7_000))
        if "ctx_length" in kwargs:
            del kwargs["ctx_length"]
        output = await self.client.chat.completions.create(messages=messages,
                                                           model=model,
                                                           stream=True,
                                                           **kwargs)
        async for chunk in output:
            # print(chunk.choices[0])
            yield chunk.choices[0].delta.content or ""
