import json
import re


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")


class KeyValue:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": ("STRING", {"default": '{"key": "This is value"}', "multiline": True}),
                "key": ("STRING", {"default": "key", "multiline": False})
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("value",)
    FUNCTION = "key_value"
    CATEGORY = 'ComfyUI-Light-Tool/DataProcessing'
    DESCRIPTION = "Get values from JSON string"

    @staticmethod
    def key_value(data, key):
        try:
            json_data = json.loads(data)
        except json.JSONDecodeError:
            return ''

        value = json_data
        parts = key.split('.')
        for part in parts:
            key_part = re.match(r'^([^\[]*)', part).group(1)
            indices = re.findall(r'\[(\d+)\]', part)

            if key_part:
                if isinstance(value, dict):
                    value = value.get(key_part, '')
                else:
                    return ''
                if value == '':
                    return ''

            for index_str in indices:
                if isinstance(value, list):
                    try:
                        index = int(index_str)
                    except ValueError:
                        return ''
                    if 0 <= index < len(value):
                        value = value[index]
                    else:
                        return ''
                else:
                    return ''
        return (value if value != '' else '',)


class SerializeJsonObject:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_object": (any_type, {"default": json.loads('{"key": "This is value"}'), "defaultInput": True}),
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("json_str",)
    FUNCTION = "get_json_str"
    CATEGORY = 'ComfyUI-Light-Tool/DataProcessing'
    DESCRIPTION = "Convert a JSON object to a JSON string"

    @staticmethod
    def get_json_str(json_object):
        try:
            json_str = json.dumps(json_object, ensure_ascii=False)
        except json.JSONDecodeError:
            return '{}'
        return (json_str,)


class DeserializeJsonString:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_str": ("STRING", {"default": '{"key": "This is value"}', "multiline": True}),
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("json_object",)
    FUNCTION = "get_json"
    CATEGORY = 'ComfyUI-Light-Tool/DataProcessing'
    DESCRIPTION = "Convert a JSON string to a JSON object"

    @staticmethod
    def get_json(json_str):
        try:
            json_dict = json.loads(json_str)
            return (json_dict,)
        except json.JSONDecodeError:
            json_dict = {}
        return (json_dict,)


NODE_CLASS_MAPPINGS = {

    "Light-Tool: KeyValue": KeyValue,
    "Light-Tool: SerializeJsonObject": SerializeJsonObject,
    "Light-Tool: DeserializeJsonString": DeserializeJsonString,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Light-Tool: KeyValue": "Light-Tool: Get values from JSON",
    "Light-Tool: SerializeJsonObject": "Light-Tool: Serialize a JSON object",
    "Light-Tool: DeserializeJsonString": "Light-Tool: Deserialize a JSON string",
}
