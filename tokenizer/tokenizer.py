from transformers import PreTrainedTokenizerFast


def load_tokenizer(tokenizer_file):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    tokenizer.pad_token = '[PAD]'
    tokenizer.cls_token = '[CLS]'
    tokenizer.sep_token = '[SEP]'
    return tokenizer