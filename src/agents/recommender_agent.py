def recommend(desc, classification, memory, llm):
    return {"recommendations": [
        {"id":"r1","type":"layer","title":"Add sub-sine","short_description":"Add clean sub.","actionable_parameters":{"synth":"sine","gain_db":-6},"confidence":0.9}
    ]}
