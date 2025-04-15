import re
import json
from functools import wraps
from langdetect import detect
from deep_translator import GoogleTranslator


cache = {}

LANG_MAP = {
    "zh-cn": "zh-CN",
    "zh-tw": "zh-TW"
}


def preprocess_text(text):
    text = text.strip().lower()
    return re.sub(r'[^\u3131-\u3163\uac00-\ud7a3\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFFa-z0-9\s]', '', text)


def get_cache():
    return cache


def cache_translation(func):
    @wraps(func)
    def wrapper(text:str, target_lang='en'):
        text = preprocess_text(text)
        key = f'{text}|{target_lang}'

        if not text.strip():
            return text
        if key in cache:
            return cache[key]

        try:
            result = func(text, target_lang)
        except Exception:
            result = text

        cache[key] = result
        return result

    return wrapper


@cache_translation
def translate(text:str, target_lang='en') -> str:
    if not re.search(r'[\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7A3]', text):
        return text

    try:
        lang = detect(text)
        lang = LANG_MAP.get(lang, lang)

        if lang == target_lang:
            return text

        translated = GoogleTranslator(source=lang, target=target_lang).translate(text)
        if re.search(r'[\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7A3]', translated):
             return text

        print(f"[translate] '{text}' to '{translated}'")
        return translated

    except Exception as e:
        print(f'lang detection/translation failed: {e} | text: {text}')
        raise


def save_cache(path='cache.json'):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(get_cache(), f, ensure_ascii=False, indent=2)
    print(f'[cache] saved {len(cache)} entries to {path}')


def load_cache(path='cache.json'):
    global cache
    try:
        with open(path, 'r', encoding='utf-8') as f:
            cache = json.load(f)
            print(f'[cache] loaded {len(cache)} entries from {path}')
    except FileNotFoundError:
        cache = {}
