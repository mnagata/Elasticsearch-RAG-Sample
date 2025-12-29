# langchain-sample

Elasticsearch と OpenAI を使用した RAG サンプル

### .env ファイル設定

```
# Elasticsearchインデックス名
ELASTIC_INDEX_XML="legal_data"
# Elasticsearchホスト名
ELASTIC_HOST="http://localhost:9200"

# OpenAI API Key
OPENAI_API_KEY="<API KEY>"

# GPTモデル
GPT_MODEL="gpt-3.5-turbo-0125"

# XMLドキュメントパス
XML_DOCUMENT_PATH="<path>"

# Embeddingに使用するモデル保存先
MODEL_DIR="models"
# Embeddingに使用するモデル名
MODEL_NAME="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
```

### Elasticsearch 起動

開発用のためセキュリティ系は OFF で起動

```
$ docker-compose -f docker-compose.dev.yml up -d
```

### Elasticsearch 起動確認

```
$ curl http://localhost:9200/
{
  "name" : "elasticsearch",
  "cluster_name" : "es-docker-cluster",
  "cluster_uuid" : "SBMSTSUTTqyryJtW1CCk5w",
  "version" : {
    "number" : "8.12.0",
    "build_flavor" : "default",
    "build_type" : "docker",
    "build_hash" : "1665f706fd9354802c02146c1e6b5c0fbcddfbc9",
    "build_date" : "2024-01-11T10:05:27.953830042Z",
    "build_snapshot" : false,
    "lucene_version" : "9.9.1",
    "minimum_wire_compatibility_version" : "7.17.0",
    "minimum_index_compatibility_version" : "7.0.0"
  },
  "tagline" : "You Know, for Search"
}
```

### モデルをダウンロード

ローカルにモデルデータをダウンロード

```
$ python hf_model_downloader.py
```

### ドキュメント読み込み

法令データをダウンロードして展開  
https://elaws.e-gov.go.jp/download/

```
$ python legal_doc_loader.with_hf.py
```

### Q&A

読み込んだ法令を使って Q&A の実行

```
$ python question_hf.py

＞ 質問を入力してください (終了するには 'exit' と入力):
森林に関する法令をリストアップしてください
--- Answer ---
1. 森林法（昭和二十六年法律第二百四十九号）
2. 森林・林業基本法（昭和三十九年法律第百六十一号）
3. 森林・林業基本計画

これらの法令は森林や林業に関する施策や計画を定めています。

＞ 質問を入力してください (終了するには 'exit' と入力):
森林・林業基本法の概要を教えてください
--- Answer ---
森林・林業基本法は森林及び林業に関する施策についての基本理念及び責務を定め、森林及び林業に関する施策を総合的かつ計画的に推進し、国民生活の安定向上及び国民経済の健全な発展を目的としています


質問を入力してください (終了するには 'exit' と入力):
exit
プログラムを終了します。
```
