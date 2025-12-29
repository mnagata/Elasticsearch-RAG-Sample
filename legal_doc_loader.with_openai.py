import pathlib
import os
import time
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredXMLLoader
from langchain_community.vectorstores.elasticsearch import ElasticsearchStore

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ELASTIC_INDEX = os.environ["ELASTIC_INDEX"]
ELASTIC_USER = os.environ["ELASTIC_USER"]
ELASTIC_PASSWORD = os.environ["ELASTIC_PASSWORD"]
ELASTIC_HOST = os.environ["ELASTIC_HOST"]
XML_DOCUMENT_PATH = os.environ["DXML_DOCUMENT_PATH"]


# ssl_verify = {
#    "verify_certs": True,
#    "basic_auth": (ELASTIC_USER, ELASTIC_PASSWORD),
#    "ca_certs": "./http_ca.crt",
# }
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

xml_doc_path = pathlib.Path(XML_DOCUMENT_PATH)

xml_file_paths = []

for xml_path in xml_doc_path.glob("**/*.xml"):
    xml_file_paths.append(str(xml_path))

# 読み込む件数の制限
xml_file_paths = xml_file_paths[-30:]

cnt = 0

for xml_file in xml_file_paths:
    loader = UnstructuredXMLLoader(xml_file)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)

    split_count = len(split_docs)
    cnt = cnt + 1
    print(str(cnt) + " : " + str(split_count) + " : " + xml_file)

    for split_doc in split_docs:
        ElasticsearchStore.from_documents(
            [split_doc],
            embedding=embeddings,
            es_url=ELASTIC_HOST,
            es_user=ELASTIC_USER,
            es_password=ELASTIC_PASSWORD,
            index_name=ELASTIC_INDEX,
            strategy=ElasticsearchStore.ApproxRetrievalStrategy(),
        )
        # OpenAI API tokens per min(TPM) エラー対策のための時間待ち
        time.sleep(0.25)
