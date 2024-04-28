from datetime import datetime
import os

from openai import OpenAI
from pydantic.v1 import BaseModel

from scripts.agent import OpenAIAgent
from scripts.tool import Tool



# === TOOL DEFINITIONS ===

class Expense(BaseModel):
    description: str
    net_amount: float
    gross_amount: float
    tax_rate: float
    date: datetime


def add_expense_func(**kwargs):
    return f"Added expense: {kwargs} to the database."


add_expense_tool = Tool(
    name="add_expense_tool",
    model=Expense,
    function=add_expense_func
)


class ReportTool(BaseModel):
    report: str = None


def report_func(report: str = None):
    return f"Reported: {report}"


report_tool = Tool(
    name="report_tool",
    model=ReportTool,
    function=report_func
)


class DateTool(BaseModel):
    x: str = None


get_date_tool = Tool(
    name="get_current_date",
    model=DateTool,
    function=lambda: datetime.now().strftime("%Y-%m-%d"),
    validate_missing=False
)

tools = [
    add_expense_tool,
    report_tool,
    get_date_tool
]


# === RUN AGENT ===

client = OpenAI(api_key=os.getenv("API_KEY"))
model_name = "gpt-3.5-turbo-0125"
agent = OpenAIAgent(tools, client, model_name=model_name, verbose=True)

user_input = "I have spend 5$ on a coffee today please track my expense. The tax rate is 0.2"

agent.run(user_input)
