import json
import re

class ZeroEgressSanitizer:
    @staticmethod
    def sanitize_graph_json(graph_str: str) -> str:
        clean_str = re.sub(r'/(?:[a-zA-Z0-9_\-\.]+/)+[a-zA-Z0-9_\-\.]+', '<REDACTED_PATH>', graph_str)
        try:
            data = json.loads(clean_str)
            return json.dumps(data, indent=2)
        except Exception:
            return clean_str
