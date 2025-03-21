import json


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
                "data": ("STRING", {"default": '{"key": "This is value"}', "multiline": False}),
                "key": ("STRING", {"default": "key", "multiline": False})
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("value",)
    FUNCTION = "key_value"
    CATEGORY = 'ComfyUI-Light-Tool/DataProcessing'
    DESCRIPTION = "Get values from JSON"

    @staticmethod
    def key_value(data, key):
        json_data = json.loads(data)
        value = json_data.get(key, '')
        return (value, )


NODE_CLASS_MAPPINGS = {

    "Light-Tool: KeyValue": KeyValue
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Light-Tool: KeyValue": "Light-Tool: Get values from JSON",
}