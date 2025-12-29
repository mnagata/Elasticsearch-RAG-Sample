if [ ! -e '/check' ]; then
    touch /check
    # 初回起動時に実行させたいコマンドをここに書く
    bin/elasticsearch-plugin install analysis-icu
    bin/elasticsearch-plugin install analysis-kuromoji
    echo "セットアップ"
else
    # 2回目以降
    echo "セットアップ済"
fi