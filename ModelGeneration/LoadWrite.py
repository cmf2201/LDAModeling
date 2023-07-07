import json

# file sole purpose to house these functions
# load data
def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# write data
def write_data(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
