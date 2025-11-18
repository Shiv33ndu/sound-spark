from src.tools.feature_extractor import compute_basic_descriptors
from google.adk.agents import SequentialAgent, LlmAgent, Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search, AgentTool, load_memory
from google.adk.tools.function_tool import FunctionTool
from google.adk.apps.app import App, ResumabilityConfig, EventsCompactionConfig
from google.adk.tools.tool_context import ToolContext
from utils.retry_config import retry_config
import warnings

from utils.output_schema import ClassificationOutput
from src.tools.mcp_sound_tool import mcp_sound_server

warnings.filterwarnings("ignore")

# ================================================
# 1 Audio Feature Extracter tool 
_feature_agent_instance = Agent(
    name="feature_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config), 
    description="A simple agent that can describe the given audio sample.",
    instruction="""
    You are SoundSpark a creative music producer assistant, with broad knowledge of sound designing and muic production and also equipped with audio feature extraction capabilities.
    1. The user will provide a prompt containing a file path.
    2. Use the 'compute_basic_descriptors' tool with that 'audio_path'.
    3. Your final output MUST be a valid JSON object with a SINGLE key 
       named 'descriptor'.
    """,
    tools=[compute_basic_descriptors],
    output_key="descriptors"
)

print("[orchestrator] : Audio feature agent created")

# ==================================================







# ==================================================
# 2 Classifier agent, to classify the genre, mood of the given audio sample 
_classifier_agent_instance = Agent(
    name="classifier_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    You are an expert audio classifier.
    
    1. You will receive JSON string {descriptors} containing audio features.
    2. Analyze these features to determine the nature of the sound.
    3. STRICTLY Return JSON with keys: 
        "style_tags" (list of up to 4 descriptive tags), 
        "genre_suggestions" (list up to 3), 
        "texture" (one of: 'gritty','warm','bright','dark','percussive','smooth','wide'), 
        "confidence" (0-1 float). 
    """,
    output_schema=ClassificationOutput,
    output_key="classification",
)

print("[orchestrator] : Audio Classifier agent created")

# ==================================================







# ===================================================

# 3. Recommender agent, 
_recommender_agent = Agent(
    name="recommender_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a sound recommender, who have professional and creative knowledge about sound designing and musical genres.
    1. use the {descriptors} and {classification} information of the audio given by the user and user prompt
    2. if user prompt has an intent or goal to do with given sound use that to give recommendations of the sounds or you can use your own creative approach 
    3. Style rules snippet (JSON) which lists typical layers, fx_chains, sample_keywords and preset_tweaks for the detected style.
    
    Produce a JSON object: {{
        "recommendations": [
            {{ "id": "<id>", "type":"layer|fx_chain|preset_tweak|sample_keyword|variation",
            "title":"", "short_description":"", "actionable_parameters":{{}}, "confidence":0.0 }}
        ]
        }}

    Constraints:
    - Produce 4 recommendations, ranked by confidence (highest first).
    - For each "actionable_parameters" include concrete parameters (e.g. cutoff_hz, gain_db, synth: 'sine', filter: {{...}}).
    - Output MUST BE a JSON
    """,
    output_key="recommendations",
) 

print("[orchestrator] : Recommender Agent is created")
# ===================================================



# ===================================================

# 4. sample search agent
_sample_search_agent = Agent(
    name="sample_search_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="This agent will rely on recommendations and search for such sounds and show it to users to preview it",
    instruction="""You are a sample sound searcher
    - using the type layer in {recommendations}, use the tool 'mcp_sound_server' to look for 5 distict sounds to recommend to user
    - lit the 5 found sounds in below manner 
        - found sound sample name : it's preview URL IMPORTATN! THE URL COMES AFTER SOUND NAME AND ALL URLs MUST BE WORKING ONES  
    """,
    tools=[mcp_sound_server],
    output_key='preview_sounds'
)
# ===================================================



# root agent to do the talking with user and aggregation of the things
_seggregator_agent = Agent(
    name="seggregator_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="This is the main face of the sound designer agent, it will manage other subagents too and pass them the key info needed",
    instruction="""
    Combine these three results into one response
    - {classification} turn this JSON into bullet point, consider Top level to be parent and do indentation for childs bullet points, REDACT CONFIDENCE VALUE
    - {recommendations} turn this JSON into bullet point, consider Top level to be parent and do indentation for childs bullet points REDACT CONFIDENCE VALUE & DON'T REWRITE 'short_description'
    - {preview_sounds}, DON'T REWRITE AND PRESERVE THE STRUCTURE
    - All points MUST be one liner
    """,
)
   

print("[orchestrator] : Root agent to use orchestrator is created")



# Orchestrator pipeline 
orchestrator = SequentialAgent(
    name= "orchestrator",
    description="This is the start of the pipeline that orchestrates and runs the sub-agents in the sequential manner.",
    sub_agents=[_feature_agent_instance, _classifier_agent_instance, _recommender_agent, _sample_search_agent, _seggregator_agent], 
)

print("[orchestrator] : Orchestrator  Pipeline created")



# Orchestrator App Wrapper for advanced feature access
orchestrator_app = App(
    name="agents",
    root_agent=orchestrator,   # TODO : we need to replace orchestrator with an agent that can take these values and work on them, root agent is messing up.
    resumability_config=ResumabilityConfig(is_resumable=True),
)

#  =================================================================================================================
#  We need a light pipeline route, in case user didn't share or wants any audio file analysis and suggestion on that 
#  =================================================================================================================

# lighter chat_agent without any heavy tools and MCP calls and with low latency to reply to user   
chat_agent = LlmAgent(
    name="chat_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    description="This agent deals with a scenario where user don't need any file analysis or have sound designing sample suggestions user might ask like: what was my previously sent file, It will simply do the database Session look up for this instead of redoing the orchestrator pipeline",
    instruction="""You are SoundSpark's conversational assistant. 
    - Answer user questions about music, sound design and recommendations use the session context.
    - if user asks about previous conversation use 'load_memory' to look for the context 
    - Do not assume or invent user's uploaded files. If the user refers to 'my previous file' and no session file exists, ask them to re-upload or provide the file path.
    - If the user asks a general chatty question unrelated to audio (e.g., 'how are you', 'what plugins are good for reverb'), simply reply if unsure you may use google_search tool.
    - If unclear whether we need audio analysis, choose "clarify" and ask a 1-line clarifying question.
    """,
    tools=[google_search]  #TODO : Use load_memory tool via VertexAI
)





# Chat App wrapper in case user just gives prompt and wants no audio analysis for sound design
chat_app = App(
    name="agents",
    root_agent=chat_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
) 
