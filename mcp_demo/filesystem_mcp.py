from mcp.server.fastmcp import FastMCP
import os

mcp = FastMCP("EducosysFileSystem")

@mcp.tool()
def addFile(filename: str):
  """Create a new file in current directory"""
  if not os.path.exists(filename):
      with open(filename, "w") as f:
          pass
      print(f"File '{filename}' created.")
  else:
      print(f"File '{filename}' already exists.")


@mcp.tool()
def addFolder(directory_name: str):
  """Create a new Directory in current directory"""
  if not os.path.exists(directory_name):
      os.mkdir(directory_name)
      print(f"Directory '{directory_name}' created.")
  else:
      print(f"Directory '{directory_name}' already exists.")

if __name__ == "__main__":
  mcp.run(transport="stdio")
