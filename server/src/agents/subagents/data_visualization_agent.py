"""Data Visualization Sub-Agent.

Collects review/rating/pricing data and generates Python code
for visualizations, executed via ADK's built-in code executor.
"""

from google.adk.agents import LlmAgent
from google.adk.code_executors import BuiltInCodeExecutor
import os

from agents.prompts.visualization_prompt import VISUALIZATION_INSTRUCTION

data_visualization_agent = LlmAgent(
    name="data_visualization_agent",
    model=os.getenv("DEFAULT_MODEL", "qwen3-vl:235b-cloud"),
    instruction=VISUALIZATION_INSTRUCTION,
    description=(
        "Creates data visualizations for comparing hotels, restaurants, flights, "
        "and trip budget breakdowns. Generates and executes Python code (matplotlib) "
        "to produce charts and graphs for informed decision-making."
    ),
    tools=[],
    code_executor=BuiltInCodeExecutor(),
)
