from src.tools.feature_extractor import compute_basic_descriptors
class AudioFeatureAgent:
    def run(self, audio_path: str):
        return compute_basic_descriptors(audio_path)
