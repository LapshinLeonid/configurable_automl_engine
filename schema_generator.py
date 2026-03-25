import json
import os
import inspect
from configurable_automl_engine.training_engine.config_parser import Config

def clean_description(obj):
    """
    Рекурсивно очищает описания, убирая лишние отступы (indentation),
    но сохраняя структуру переноса строк.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "description" and isinstance(v, str):
                # cleandoc убирает общие отступы слева, которые
                # появляются при использовании многострочных строк в классах
                obj[k] = inspect.cleandoc(v).strip()
            else:
                clean_description(v)
    elif isinstance(obj, list):
        for item in obj:
            clean_description(item)
def save_schema(filename="config.schema.json"):
    # 1. Генерируем схему
    schema_dict = Config.model_json_schema()
    # 2. Очищаем описания, сохраняя структуру
    clean_description(schema_dict)
    abs_path = os.path.abspath(filename)
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # indent=2 отвечает за "красивое дерево" самого JSON-файла
            json.dump(schema_dict, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Схема сохранена: {abs_path}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")

save_schema()