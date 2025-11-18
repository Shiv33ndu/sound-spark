import json, numpy as np, os
from src.orchestrator import orchestrator_app, chat_app
from synthesize import synth_app, handle_llm_tool_call
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import google_search
from google.genai import types
import asyncio
from dotenv import load_dotenv
import warnings
from pathlib import Path

from utils.text_from_resp import extract_human_text
from utils.run_sessions import run_session, run_session_return
from utils.check_prompt import has_audio_path
from debug import verbose_debug_events

# This will ignore all warning messages
warnings.filterwarnings('ignore')

load_dotenv()

print("[app]: env secrets loaded!")

sample = "tests/sample_audio/gritty_bass.wav"


async def run_workflow(prompt: str):
    """
    Runs the full agent workflow with a user prompt.
    """

    #local persitent Sqlite DB to store per session related data  
    db_url = "sqlite:///memory_bank.db"
    session_service = DatabaseSessionService(db_url=db_url)
    memory_service = InMemoryMemoryService()  # TODO: replace this with Vertax AI while deploying
    
    print(f"Starting workflow for: '{prompt}'")

    # Runner with persistent storage with a check on prompt
    if has_audio_path(prompt):
        runner = Runner(app=orchestrator_app, session_service=session_service, memory_service=memory_service)
        
        print("\n--- ✅ Final Workflow Output ---")
        await run_session(runner, prompt, "test_session_01", "user_01")
        
        print("\n--- Creating Demo Synthesized Sound Ouput ---")
        runner_2 = Runner(app=synth_app, session_service=session_service, memory_service=memory_service)
        
        resp = await run_session_return(runner_2, prompt, "test_session_01", "user_01")

        ok = handle_llm_tool_call(resp, sample, f"tests/synthesis_demo/{Path(sample).stem}_layered.mp3")
        
        print(ok)
        
        # adding the session to long term memory
        print("\n[app]: Adding the session to long term memory")

        session = await runner.session_service.get_session(app_name=runner.app_name, user_id="user_01", session_id="test_session_01")
        synth_sess = await runner_2.session_service.get_session(app_name=runner_2.app_name, user_id="user_01", session_id="test_session_01")
        await memory_service.add_session_to_memory(session) 
        await memory_service.add_session_to_memory(synth_sess)

        print("[app]: Memory Saved!\n\n")

        mem = await memory_service.search_memory(app_name=runner.app_name, user_id="user_01", query="What sound we used")
        
        for memory in mem.memories:
            if memory.content and memory.content.parts:
                text = memory.content.parts[0].text[:80]
                print(f"  [{memory.author}]: {text}...")

        mem2 = await memory_service.search_memory(app_name=runner_2.app_name, user_id="user_01", query="What sound synthesis")
        
        for memory in mem2.memories:
            if memory.content and memory.content.parts:
                text = memory.content.parts[0].text[:80]
                print(f"  [{memory.author}]: {text}...")

    else:
        runner = Runner(app=chat_app, session_service=session_service, memory_service=memory_service)
        print("\n--- ✅ Final Workflow Output ---")
        await run_session(runner, prompt, "test_session_01", "user_01")
    # response = await runner.run_debug(prompt)

    
    
    # print("\n--- ✅ Debug Workflow Output ---")
    # # await verbose_debug_events(runner, prompt, "test_session_01", "user_01")
    
    # print(extract_human_text(response))
    print("\n\n")
    # print(response)

if __name__ == "__main__":
    # Get a real file path for testing
    # Note: Replace with a real audio file path on your system
    user_prompt = f"what sounds to layer with this audio {sample}"  # what sounds to layer with this audio {sample}

    asyncio.run(run_workflow(user_prompt))