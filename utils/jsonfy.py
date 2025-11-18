import json
import re

def give_json(res: str):
    """
    A utility to structure the text false JSON string from LLMs to JSON String
    This eliminates any texts or words out side of the JSON string parantheses 

    args:
        res : string, LLMs faulty response
    return
        JSON Strnig Dict in Python 
    """
    # Use a regex to extract the content between the first { and the last }
    json_match = re.search(r'\{.*\}', res, re.DOTALL)
    
    if json_match:
        # If a JSON-like object is found, use that for parsing
        json_string = json_match.group(0)
    else:
        # Fallback to the raw string if no JSON is found
        json_string = res
        
    try:
        response = json.loads(json_string)
        return response
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from LLM: {e}")
        print("Raw LLM response:")
        print(repr(res)) # Using repr() shows hidden characters like newlines
        return None