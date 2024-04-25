import colorama
import json
from colorama import Fore
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import OpenAI
from pydantic.v1 import BaseModel
from datetime import datetime
from typing import Type, Callable


class ToolResult(BaseModel):
    content: str
    success: bool


class Tool(BaseModel):
    name: str
    model: Type[BaseModel]
    function: Callable
    validate_missing: bool = True

    class Config:
        arbitrary_types_allowed = True

    def run(self, **kwargs) -> ToolResult:
        if self.validate_missing:
            missing_values = self.validate_input(**kwargs)
            if missing_values:
                content = f"Missing values: {', '.join(missing_values)}"
                return ToolResult(content=content, success=False)

        result = self.function(**kwargs)
        return ToolResult(content=str(result), success=True)

    def validate_input(self, **kwargs):
        missing_values = []

        for key, value in self.model.__annotations__.items():
            if key not in kwargs:
                missing_values.append(key)

        return missing_values

    @property
    def openai_tool_schema(self):
        schema = convert_to_openai_tool(self.model)
        # set function name
        schema["function"]["name"] = self.name

        # remove required field
        if schema["function"]["parameters"].get("required"):
            del schema["function"]["parameters"]["required"]
        return schema