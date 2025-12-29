import os
import pathlib
import datetime
import splitter
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores.elasticsearch import ElasticsearchStore
from langchain_community.embeddings import HuggingFaceEmbeddings


def main():
    env = load_environ()

    document_load(env)


def load_environ():
    """環境変数ロード

    Returns:
        dict[str, str]: 環境変数dict
    """
    load_dotenv()

    env = {
        "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
        "GPT_MODEL": os.environ["GPT_MODEL"],
        "MODEL_DIR": os.environ["MODEL_DIR"],
        "MODEL_NAME": os.environ["MODEL_NAME"],
        "ELASTIC_INDEX_PDF": os.environ["ELASTIC_INDEX_PDF"],
        "ELASTIC_USER": os.environ["ELASTIC_USER"],
        "ELASTIC_PASSWORD": os.environ["ELASTIC_PASSWORD"],
        "ELASTIC_HOST": os.environ["ELASTIC_HOST"],
        "PDF_DOCUMENT_PATH": os.environ["PDF_DOCUMENT_PATH"],
    }
    return env


def document_load(env):
    """ドキュメントロード
    ベクターストアにPDF文書を格納

    Args:
        env (dict): 環境変数
    """
    model_path = f"{env['MODEL_DIR']}/{env['MODEL_NAME']}"
    hf = HuggingFaceEmbeddings(model_name=model_path)

    pdf_doc_path = pathlib.Path(env["PDF_DOCUMENT_PATH"])
    pdf_file_paths = []

    for pdf_path in pdf_doc_path.glob("**/*.pdf"):
        pdf_file_paths.append(str(pdf_path))

    cnt = 0
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, "JST")
    now = datetime.datetime.now(JST)
    print(now)

    for pdf_file in pdf_file_paths:
        loader = UnstructuredPDFLoader(
            file_path=pdf_file,
            mode="single",  # {'elements', 'single', 'paged'}
            strategy="fast",  # {'hi_res', 'fast'}
        )
        docs = loader.load()

        text_splitter = splitter.recursive_character()
        chunk = text_splitter.split_documents(docs)

        split_count = len(chunk)
        cnt = cnt + 1
        print(str(cnt) + " : " + str(split_count) + " : " + pdf_file)

        db = ElasticsearchStore.from_documents(
            chunk,
            embedding=hf,
            es_url=env["ELASTIC_HOST"],
            es_user=env["ELASTIC_USER"],
            es_password=env["ELASTIC_PASSWORD"],
            index_name=env["ELASTIC_INDEX_PDF"],
            strategy=ElasticsearchStore.ApproxRetrievalStrategy(hybrid=True),
        )
        db.client.indices.refresh(index=env["ELASTIC_INDEX_PDF"])

    now = datetime.datetime.now(JST)
    print(now)


if __name__ == "__main__":
    main()
