import json
import os

FILE = "history.json"

class PredictionHistory:
    def __init__(self):
        # Create empty file if not exists
        if not os.path.exists(FILE):
            with open(FILE, "w") as f:
                json.dump([], f)

    def add(self, ticker, signal, confidence):
        data = self.get_all()

        # Ensure confidence is a float
        try:
            confidence_text = f"{float(confidence):.2f}%"
        except:
            confidence_text = str(confidence)

        data.append([ticker.upper(), signal, confidence_text])

        with open(FILE, "w") as f:
            json.dump(data, f, indent=4)

    def get_all(self):
        try:
            with open(FILE, "r") as f:
                data = json.load(f)

            # Ensure file content is always list
            if not isinstance(data, list):
                return []

            return data

        except json.JSONDecodeError:
            # Fix corrupted file
            with open(FILE, "w") as f:
                json.dump([], f)
            return []

        except:
            return []
