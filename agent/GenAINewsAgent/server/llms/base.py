from abc import ABC, abstractmethod
from typing import List, Dict, Union


class BaseLLM(ABC):

    def __init__(self, api_key: Union[str, None] = None, **kwargs):
        self.api_key = api_key
        self.client = None
        self.extra_args = kwargs

    @abstractmethod
    async def __call__(self, model: str, messages: List[Dict], **kwargs):
        pass
