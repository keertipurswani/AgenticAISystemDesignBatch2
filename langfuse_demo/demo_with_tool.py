from dotenv import load_dotenv
load_dotenv()

import subprocess

from langchain.agents import create_agent
from langchain.tools import tool

from langfuse import get_client
from langfuse.langchain import CallbackHandler


# Initialize Langfuse
langfuse = get_client()
langfuse_handler = CallbackHandler()


# --------------------------------------------------
# Terminal Tool
# --------------------------------------------------

_TIME_OUT_SECONDS = 30


@tool
def run_command(command: str) -> str:
    """Run a shell command and return its output."""

    if not command or not command.strip():
        return "Error: command cannot be empty"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=_TIME_OUT_SECONDS,
        )

        if result.returncode != 0:
            return f"Error: {result.stderr}"

        return result.stdout

    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {_TIME_OUT_SECONDS} seconds"

    except Exception as e:
        return f"Error: {e}"


# --------------------------------------------------
# Agent
# --------------------------------------------------

def run_agent():

    agent = create_agent(
        "openai:gpt-4o",
        tools=[run_command],
        system_prompt="""
        You are a helpful coding assistant.

        You have access to a terminal tool that can execute
        shell commands.

        Use the terminal whenever the user asks about:
        - files in the current directory
        - Python version
        - installed packages
        - system information
        - running commands

        Explain the result clearly to the user.
        """
    )

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What Python version am I using?"
                }
            ]
        },
        config={
            "callbacks": [langfuse_handler]
        }
    )

    print(response["messages"][-1].content)


if __name__ == "__main__":
    run_agent()