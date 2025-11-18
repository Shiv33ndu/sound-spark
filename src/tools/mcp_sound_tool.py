from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams

import os
import asyncio
from dotenv import load_dotenv

from utils.mcp_wakeup import wait_for_wakeup
from utils.run_sessions import run_session


# load_dotenv()

print("[MCP_TOOL] : ADK components imported successfully.")


freesound_api_key = os.getenv("FREESOUND_API_KEY", "")
mcp_server_uri = os.getenv("MCP_SERVER_URI", "https://freesound-mcp-server.onrender.com")+"/mcp"


print("[MCP_TOOL] : Checking the MCP server status")

ok = asyncio.run(wait_for_wakeup(mcp_server_uri, on_wakeup_message=print))

if ok:
    print("[MCP_TOOL]: Server woke up!!")
else:
    print("[MCP_TOOL]: Server did NOT wake up in time")


# MCP integration with Everything Server
mcp_sound_server = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url= mcp_server_uri,
        headers= {"Authorization": freesound_api_key},
    ),
)

print("[MCP_TOOL]: MCP Tool created")





