import os
from langchain import hub
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores.elasticsearch import ElasticsearchStore

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ELASTIC_INDEX = os.environ["ELASTIC_INDEX"]
ELASTIC_USER = os.environ["ELASTIC_USER"]
ELASTIC_PASSWORD = os.environ["ELASTIC_PASSWORD"]
ELASTIC_HOST = os.environ["ELASTIC_HOST"]

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

vectorstore = ElasticsearchStore(
    es_url=ELASTIC_HOST,
    es_user=ELASTIC_USER,
    es_password=ELASTIC_PASSWORD,
    index_name=ELASTIC_INDEX,
    embedding=embeddings,
    strategy=ElasticsearchStore.ApproxRetrievalStrategy(),
)

retriever = vectorstore.as_retriever()
# prompt = hub.pull("rlm/rag-prompt")
template = """あなたは質問応答タスクのアシスタントです。 取得したコンテキストの次の部分を使用して質問に答えます。 答えがわからない場合は、わからないと言ってください。 最大3つの文を使用し、回答は簡潔にしてください。:
{context}

質問: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ユーザー入力を受け付けるループ
while True:
    query = input("\n\n質問を入力してください (終了するには 'exit' と入力): \n")
    if query.lower() == "exit":
        print("プログラムを終了します。")
        break
    print("回答:")
    for chunk in rag_chain.stream(query):
        print(chunk, end="", flush=True)

# cleanup
# try:
#    vectorstore.delete_collection()
# except ValueError as e:
#    print(f"エラー: {e}")
