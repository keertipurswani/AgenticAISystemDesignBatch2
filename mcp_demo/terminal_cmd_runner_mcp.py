# terminal_cmd_runner_mcp.py
from mcp.server.fastmcp import FastMCP
import subprocess


mcp = FastMCP("TerminalCmdRunner")


@mcp.tool()
def run_command(command: str) -> str:
   """Run a shell command and return its output"""
   result = subprocess.run(
       command,
       shell=True,
       capture_output=True,
       text=True
   )
   output = result.stdout
   if result.stderr:
       output += f"\nSTDERR: {result.stderr}"
   return output or "(no output)"


@mcp.tool()
def run_in_directory(command: str, directory: str) -> str:
   """Run a shell command inside a specific directory"""
   result = subprocess.run(
       command,
       shell=True,
       capture_output=True,
       text=True,
       cwd=directory
   )
   output = result.stdout
   if result.stderr:
       output += f"\nSTDERR: {result.stderr}"
   return output or "(no output)"


if __name__ == "__main__":
   mcp.run(transport="stdio")
