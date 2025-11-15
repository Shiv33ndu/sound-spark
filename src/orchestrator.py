from src.agents.audio_feature_agent import AudioFeatureAgent
from src.agents.classifier_agent import classify_descriptors
from src.agents.recommender_agent import recommend
from src.llm_client import LLMClient
from src.memory_bank import MemoryBank

class Orchestrator:
    def __init__(self):
        self.llm = LLMClient()
        self.memory = MemoryBank()

    def run(self, audio_path, user_id="default"):
        desc = AudioFeatureAgent().run(audio_path)
        cls = classify_descriptors(desc, self.llm)
        mem = self.memory.load_user(user_id)
        recs = recommend(desc, cls, mem, self.llm)
        return {"descriptors": desc, "classification": cls, "recommendations": recs}
