from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.apps.app import App, ResumabilityConfig

from src.tools.code_exec_tool import execute_tool

import json
from typing import Dict, Any



# tool consume LLM output for synthesizing the audio
def handle_llm_tool_call(llm_json_text: str, file_path: str, out_path: str) -> Dict[str, Any]:
    """
    This function takes the LLM Response (assumes its JSON STRING)
    extracts function and args key out of it 
    calls execute_tool that does extra validation on the instructions part of the LLM response
    down the pipeline apply_patch fn gets called to create a synthesized sound and store it into local dir

    arg:
        llm_json_text : JSON String that has all the suggestions and instructions to build the patch
        file_path: uploaded file's path
        out_path: syntesized file's path
    return:
        JSON 
    """
    call = json.loads(llm_json_text)
    if call.get("tool") != "synthesis_tool":
        return {"ok": False, "error": "unsupported tool"}
    func = call.get("function")
    args = call.get("args", {})
    resp = execute_tool(func, args, file_path, out_path)
    return resp


synth_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite"),
    name="synth_agent",
    instruction="""You are a sound creative synthesizer agent
    - You take user prompt and the audio sample via audio_path
    - if user has any intent for the sound try to choose fx process for that, otherwise you are free to create your own fx chain
    - example: {
                "tool": "synthesis_tool",
                "function": "apply_patch",
                "args": {
                    "input_audio_path": "relative path",
                    "out_path": "relative path",
                    "sr": 22050,
                    "mix_ratio": 0.75,
                    "params": { ... }        // structured parameters (MUST)
                    }
                }
    - Rules & constraints:
        1. Types: Use proper JSON types — numbers should be numbers (not strings), booleans true/false, arrays for lists, objects for maps.
        2. If you include an instruction string, it will be parsed; **prefer** returning the structured `params` object (more reliable).
        3. Allowed `params` keys (optional; include only those required): 
            - "sub_sine": {"enabled": bool, "freq_hz": number, "amp": number (0-1), "lowpass_cutoff": number (hz) }
            - "noise": {"enabled": bool, "amp": number}
            - "distortion": {"enabled": bool, "drive": number}
            - "delay": {"enabled": bool, "ms": integer, "feedback": number (0-1)}
            - "global_lowpass": number (hz)
            - "global_highpass": number (hz)
        4. `mix_ratio` (0-1): proportion of original audio in final mix. Use values like 0.6, 0.75.
        5. Keep numeric values realistic (Hz frequencies typically 20-20000, delay ms 10-600, amp 0-1, drive 0.5-3).
        6. If uncertain about exact numbers, pick conservative defaults that produce musical results (e.g., sub_sine amp 0.4-0.6, lowpass 120Hz).
        7. If you cannot find a clear param mapping, include a conservative default `params` object with `sub_sine.enabled = true` and sensible defaults.
        8. **Do not** request arbitrary code execution or unvalidated paths.
        9. Output must parse as JSON with no extra characters.
    """,
)

synth_app = App(
    name="synth_app",
    root_agent=synth_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)

# async def runit():
    
#     # sample = "tests/sample_audio/pluck.wav"

#     # runner = InMemoryRunner(agent=sound_search)
    
#     # response = await runner.run_debug(f"Do something creative sound-designing with this sound {sample}")

#     # print(response)

#     inst = """
#         {
#             "tool": "synthesis_tool",
#             "function": "apply_patch",
#             "args": {
                # "input_audio_path": "tests/sample_audio/gritty_bass.wav",
                # "out_path": "tests/synthesis_demo/gritty_bass_processed.wav",
#                 "sr": 44100,
#                 "mix_ratio": 0.6,
#                 "params": {
#                     "sub_sine": {
#                         "enabled": true,
#                         "freq_hz": 60,
#                         "amp": 0.5,
#                         "lowpass_cutoff": 150
#                     },
#                     "distortion": {
#                         "enabled": true,
#                         "drive": 2.0
#                     },
#                     "noise": {
#                         "enabled": false
#                     },
#                     "delay": {
#                         "enabled": false
#                     },
#                     "global_lowpass": 800,
#                     "global_highpass": 30
#                 }
#             }
#         }
# """
    
#     inst2 = """
#     {
#     "tool": "synthesis_tool",
#     "function": "apply_patch",
#     "args": {
#         "input_audio_path": "tests/sample_audio/gritty_bass.wav",
#         "out_path": "tests/synthesis_demo/gritty_bass_melodic.wav",
#         "sr": 44100,
#         "mix_ratio": 0.7,
#         "params": {
#             "sub_sine": {
#                 "enabled": true,
#                 "freq_hz": 110,
#                 "amp": 0.5,
#                 "lowpass_cutoff": 300
#             }
#         }
#     }
# }
#     """
    
#     inst3 = """
#     {
#     "tool": "synthesis_tool",
#     "function": "apply_patch",
#     "args": {
#         "input_audio_path": "tests/sample_audio/pluck.wav",
#         "out_path": "tests/synthesis_demo/pluck_creative.mp3",
#         "sr": 44100,
#         "mix_ratio": 0.7,
#         "params": {
#             "sub_sine": {
#                 "enabled": true,
#                 "freq_hz": 80,
#                 "amp": 0.5,
#                 "lowpass_cutoff": 200
#             },
#             "distortion": {
#                 "enabled": true,
#                 "drive": 1.5
#             },
#             "delay": {
#                 "enabled": true,
#                 "ms": 150,
#                 "feedback": 0.4
#             },
#             "global_lowpass": 5000
#         }
#     }
# }
#     """
#     res = handle_llm_tool_call(inst3)

#     print(res)
# asyncio.run(runit())