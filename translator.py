import json
from functools import wraps
from langdetect import detect
from deep_translator import GoogleTranslator


cache = {}

LANG_MAP = {
    "zh-cn": "zh-CN",
    "zh-tw": "zh-TW"
}


def cache_translation(func):
    @wraps(func)
    def wrapper(text:str, target_lang='en'):
        key = f'{text}|{target_lang}'

        if not text.strip():
            return text

        if key in cache:
            return cache[text]

        result = func(text, target_lang)
        cache[key] = result

        return result

    return wrapper


@cache_translation
def translate(text:str, target_lang='en') -> str:
    try:
        lang = detect(text)
        lang = LANG_MAP.get(lang, lang)

        if lang == target_lang:
            return text

        translated = GoogleTranslator(source=lang, target=target_lang).translate(text)
        print(f"[translate] translating '{text}' to '{translated}'")
        return translated

    except Exception as e:
        print(f'lang detection/translation failed: {e} | text: {text}')
        cache[text] = text
        return text


def save_cache(path='cache.json'):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def load_cache(path='cache.json'):
    global cache
    try:
        with open(path, 'r', encoding='utf-8') as f:
            cache = json.load(f)
            print(f'[cache] loaded {len(cache)} entries from {path}')
    except FileNotFoundError:
        cache = {}
