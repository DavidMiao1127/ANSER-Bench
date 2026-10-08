from pathlib import Path
import yaml
class Loader:
    def get_prompt(self, name):
        return yaml.safe_load((Path(__file__).resolve().parents[1]/'prompts/finsight.yaml').read_text())[name] + '\nUse only the supplied closed corpus; all URLs must use corpus://. The report must be one JSON object containing id, answer, and retrieved_evidence. Follow the public task instruction exactly for answer formatting.'
def get_prompt_loader(*args, **kwargs): return Loader()
