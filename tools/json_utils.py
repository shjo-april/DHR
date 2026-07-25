import json

def read(filepath, encoding=None):
    return json.load(open(filepath, 'r', encoding=encoding))

def write(filepath, data, encoding=None):
    json.dump(data, open(filepath, 'w', encoding=encoding), indent='\t', ensure_ascii=False)