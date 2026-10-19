# zenzai-plus-Hazkey-Fcitx5-

Hazkeyをベースにzenzai-plusとllama.cpp内でのIn-Place TTTに対応させた。

- Zenzai有効化時に [zenzai-plus](https://huggingface.co/soichi1208/zenzai-plus) のFT済みGGUFを取得。
- 入力履歴からローカルで学習。
- 辞書登録画面から単語を学習。

## Quick start

Linux, Swift 6.1以上, CMake 3.21以上,Ninja,C/C++コンパイラ,Python 3.
Gettext,Fcitx5開発ライブラリ,Protobuf開発ライブラリ,protoc,Qt6,Widgets/Network/LinguistTools, CUDA toolkit (option)

```sh
./install.sh
```

環境変数:

JOBS=n でコア数

BACKEND= はvulkan, cpuがいけるcudaが規定

PREFIX= もいける

CUDAのコンパイラ指定はCMake引数。

既存Hazkeyを置き換える。

## 保存先

- 設定：`${XDG_CONFIG_HOME:-~/.config}/hazkey`
- モデル・学習状態：`${XDG_DATA_HOME:-~/.local/share}/hazkey`
- 辞書キャッシュ：`${XDG_CACHE_HOME:-~/.cache}/hazkey`

# 

## ライセンス

このフォークで独自に追加・変更した部分は **The Unlicense**。
上流Hazkey由来の部分はMIT（[元のライセンス](LICENSE.upstream-Hazkey)）、
第三者コード・辞書はそれぞれのライセンスを維持する。
モデルは別配布のCC BY-SA 4.0。[第三者表記](THIRD_PARTY_NOTICES.md)、[元のREADME](README.upstream.md)を参照。

## TTT済みモデルのexport

設定のAIタブにある「TTT済み重みをexport…」で、最後に保存されたTTT状態を
元モデルへ組み込んだ単体GGUFを保存できる。Python 3とnumpy・tqdm・PyYAML・sentencepieceが必要。GPUは使わない。
モデルを使うのに元のスナップショットや入力履歴は不要。辞書登録データは別管理のため含まれない。
確定直後は反映に数秒かかるため、反映後にexportする。元モデル・学習状態は変更しない。

CLIでも利用できる：

```sh
python3 /usr/libexec/hazkey/export_ttt.py ~/zenzai-plus-ttt.gguf
# ソースから直接実行する場合
python3 hazkey-server/tools/export_ttt.py ~/zenzai-plus-ttt.gguf
```

カスタムモデルの場合は `--model BASE.gguf --snapshot native-ttt-XXXX.bin` を指定できる。
両者は必ず同じ学習に使用した組み合わせを指定する。既存の出力ファイルは上書きしない。
別環境ではexportしたGGUFをベースモデルとして使用する。読み込み時に過去の入力履歴を再学習させる必要はない。
個人の文章から得た適応が含まれるため、個人用の引き継ぎファイルとして扱う。
