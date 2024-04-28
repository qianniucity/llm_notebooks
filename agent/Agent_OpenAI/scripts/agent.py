import colorama
from colorama import Fore
from langchain_openai import OpenAI
from pydantic.v1 import BaseModel
from scripts.tool import Tool, ToolResult
from scripts.utils import parse_function_args, run_tool_from_response


class StepResult(BaseModel):
    event: str
    content: str
    success: bool


SYSTEM_MESSAGE = """You are tasked with completing specific objectives and must report the outcomes. At your disposal, you have a variety of tools, each specialized in performing a distinct type of task.

For successful task completion:
Thought: Consider the task at hand and determine which tool is best suited based on its capabilities and the nature of the work.

Use the report_tool with an instruction detailing the results of your work.
If you encounter an issue and cannot complete the task:

Use the report_tool to communicate the challenge or reason for the task's incompletion.
You will receive feedback based on the outcomes of each tool's task execution or explanations for any tasks that couldn't be completed. This feedback loop is crucial for addressing and resolving any issues by strategically deploying the available tools.
"""


class OpenAIAgent:

    def __init__(
            self,
            tools: list[Tool],
            client: OpenAI,
            system_message: str = SYSTEM_MESSAGE,
            model_name: str = "gpt-3.5-turbo-0125",
            max_steps: int = 5,
            verbose: bool = True
    ):
        self.tools = tools
        self.client = client
        self.model_name = model_name
        self.system_message = system_message
        self.memory = []
        self.step_history = []
        self.max_steps = max_steps
        self.verbose = verbose

    def to_console(self, tag: str, message: str, color: str = "green"):
        if self.verbose:
            color_prefix = Fore.__dict__[color.upper()]
            print(color_prefix + f"{tag}: {message}{colorama.Style.RESET_ALL}")

    def run(self, user_input: str):
        self.to_console("START", f"Starting Agent with Input: {user_input}")
        openai_tools = [tool.openai_tool_schema for tool in self.tools]
        self.step_history = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_input}
        ]

        step_result = None
        i = 0

        while i < self.max_steps:
            step_result = self.run_step(self.step_history, openai_tools)
            if step_result.event == "finish":
                break
            elif step_result.event == "error":
                self.to_console(step_result.event, step_result.content, "red")
            else:
                self.to_console(step_result.event, step_result.content, "yellow")

            i += 1

        self.to_console("Final Result", step_result.content, "green")

        return step_result.content

    def run_step(self, messages: list[dict], tools):

        # plan the next step
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools
        )

        # add message to history
        self.step_history.append(response.choices[0].message)
        # check if tool call is present
        if not response.choices[0].message.tool_calls:
            step_result = StepResult(event="Error", content="No tool calls were returned.", success=False)
            return step_result

        tool_name = response.choices[0].message.tool_calls[0].function.name
        tool_kwargs = parse_function_args(response)

        # execute the tool call
        self.to_console("Tool Call", f"Name: {tool_name}\nArgs: {tool_kwargs}", "magenta")
        tool_result = run_tool_from_response(response, tools=self.tools)
        tool_result_msg = self.tool_call_message(response, tool_result)
        self.step_history.append(tool_result_msg)

        if tool_result.success:
            step_result = StepResult(
                event="tool_result",
                content=tool_result.content,
                success=True)
        else:
            step_result = StepResult(
                event="error",
                content=tool_result.content,
                success=False
            )

        return step_result

    def tool_call_message(self, response, tool_result: ToolResult):
        tool_call = response.choices[0].message.tool_calls[0]
        return {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": tool_call.function.name,
            "content": tool_result.content,
        }
