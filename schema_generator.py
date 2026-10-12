import json
import os
import inspect
from configurable_automl_engine.training_engine.config_parser import Config

def clean_description(obj):
    """
    Recursively cleans descriptions, removing excessive indentation
    while preserving the line-break structure.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "description" and isinstance(v, str):
                # cleandoc removes the common left indentation that
                # appears when using multi-line strings in classes
                obj[k] = inspect.cleandoc(v).strip()
            else:
                clean_description(v)
    elif isinstance(obj, list):
        for item in obj:
            clean_description(item)
def save_schema(filename="config.schema.json"):
    # 1. Generate the schema
    schema_dict = Config.model_json_schema()
    # 2. Clean the descriptions, preserving the structure
    clean_description(schema_dict)
    abs_path = os.path.abspath(filename)
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # indent=2 controls the "pretty tree" structure of the JSON file
            json.dump(schema_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Schema saved: {abs_path}")
    except Exception as e:
        print(f"Error: {e}")

save_schema()