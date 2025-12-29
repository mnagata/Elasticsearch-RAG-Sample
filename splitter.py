from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

_JAPANESE_SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    "、",
    "「",
    "」",
    "！",
    "？",
    "『",
    "』",
    "（",
    "）",
    " ",
    "",
]


def recursive_character():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000, chunk_overlap=0, separators=_JAPANESE_SEPARATORS
    )
    return text_splitter


def huggingface_tokenizer():
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        encoding_name="cl100k_base",
        chunk_size=3000,
        chunk_overlap=0,
        separators=_JAPANESE_SEPARATORS,
    )
    return text_splitter


def token_text():
    text_splitter = TokenTextSplitter(chunk_size=3000, chunk_overlap=0)
    return text_splitter


def stf_token(model_path):
    text_splitter = SentenceTransformersTokenTextSplitter(
        chunk_size=3000,
        chunk_overlap=0,
        model_name=model_path,
    )
    return text_splitter
