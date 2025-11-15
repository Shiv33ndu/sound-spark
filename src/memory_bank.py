import json, os
class MemoryBank:
    def __init__(self, path="memory_bank.json"):
        self.path = path
        if not os.path.exists(path):
            with open(path,"w") as f: f.write("{}")
    def load(self):
        return json.load(open(self.path))
    def save(self, data):
        json.dump(data, open(self.path,"w"), indent=2)
    def load_user(self, user_id):
        data = self.load()
        return data.get("user_profiles", {}).get(user_id, {})
    def update_user(self, user_id, new_data):
        data = self.load()
        if "user_profiles" not in data:
            data["user_profiles"] = {}
        data["user_profiles"][user_id] = new_data
        self.save(data)
