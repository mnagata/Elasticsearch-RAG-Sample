import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores.elasticsearch import ElasticsearchStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


def main():
    env = load_environ()

    vectorstore = connect_vectorstore(env)

    chain = retrieval_chain(env, vectorstore)

    qa(chain)


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
    }
    return env


def connect_vectorstore(env):
    """ベクターストアと接続

    Args:
        env (dict): 環境変数

    Returns:
        ElasticsearchStore: esベクターストア
    """
    model_path = f"{env['MODEL_DIR']}/{env['MODEL_NAME']}"
    hf = HuggingFaceEmbeddings(model_name=model_path)

    vectorstore = ElasticsearchStore(
        es_url=env["ELASTIC_HOST"],
        es_user=env["ELASTIC_USER"],
        es_password=env["ELASTIC_PASSWORD"],
        index_name=env["ELASTIC_INDEX_PDF"],
        embedding=hf,
        strategy=ElasticsearchStore.ApproxRetrievalStrategy(),
    )

    return vectorstore


def retrieval_chain(env, vectorstore):
    """LLM およびリトリーバーからチェーンをロードする

    Args:
        env (dict): 環境変数
        vectorstore (ElasticsearchStore): ベクターストア

    Returns:
        BaseConversationalRetrievalChain: チェーン
    """
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(model_name=env["GPT_MODEL"], temperature=0)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_template()},
        condense_question_prompt=condense_qa_prompt(),
        condense_question_llm=llm,
    )

    return chain


def condense_qa_prompt():
    """生成質問のプロンプト
    LLMが質問と会話履歴を受け取って、質問の言い換え（生成質問）を行う
    """
    condense_qa_template = """
次の会話とフォローアップの質問を考慮して、フォローアップの質問を元の言語で独立した質問に言い換えます。チャット履歴がない場合は、質問を独立した質問に言い換えてください。
        
チャットの履歴:
{chat_history}
        
フォローアップの質問:
{question}
        
言い換えられた独立した質問:"""

    return PromptTemplate.from_template(condense_qa_template)


def qa_template():
    """回答プロンプト
    質問と関連情報を合わせてLLMに投げる
    """
    prompt_template = """
あなたは質問応答タスクのアシスタントです。 取得したコンテキストの次の部分を使用して質問に答えます。 答えがわからない場合は、わからないと言ってください。 最大3つの文を使用し、回答は簡潔にしてください。
{context}

質問: {question}:"""

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return prompt


def qa(chain):
    """QA実行

    Args:
        chain (BaseConversationalRetrievalChain): _description_
    """
    while True:
        query = input("\n\n＞ 質問を入力してください (終了するには 'exit' と入力): \n")
        if query.lower() == "exit":
            print("プログラムを終了します。")
            break
        print("--- Answer ---")
        response = chain.invoke({"question": query})
        print(response["answer"])
        # 参照ソースを表示
        # print("--- Source ---")
        # docs = response["source_documents"]
        # for doc in docs:
        #    print(docs)


if __name__ == "__main__":
    main()
