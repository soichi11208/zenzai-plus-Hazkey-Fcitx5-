# zinai-plus-Hazkey-Fcitx5

Hazkeyをベースに、zenzai-plusとllama.cpp内でのIn-Place TTTに対応した日本語IME。

- Zenzai有効化時に [zenzai-plus](https://huggingface.co/soichi1208/zenzai-plus) のFT済みGGUFを取得。
- 入力履歴からローカルで継続学習。推論と学習を別コンテキストで実行。
- 辞書登録画面から単語を学習。通常辞書にはnico-pixivを優遇なしで追加。
- モデル・入力履歴・個人の学習状態は配布物に含まない。

## インストール（ソースからビルド）

Linux向け。Swift 6.1以降、CMake 3.21以降、Ninja、C/C++コンパイラ、Python 3、
Gettext、Fcitx5開発ライブラリ、Protobuf開発ライブラリとprotoc、Qt6 Widgets/Network/LinguistToolsが必要（Qt 6.7以降推奨）。
CUDA版はCUDA Toolkitと対応するホストコンパイラ、NVIDIAドライバが必要。
Swiftの外部依存とモデルの取得時にインターネット接続を使用する。

```sh
./install.sh
```

既定はCUDA、`/usr`へインストール。ビルド後の配置時だけsudoを使用する。
配布物には改造済みllama.cppと変換ライブラリ、辞書を同梱しており、submoduleの取得は不要。

```sh
JOBS=8 ./install.sh
BACKEND=vulkan ./install.sh  # Vulkan開発環境が必要
BACKEND=cpu ./install.sh
PREFIX=/usr/local ./install.sh
```

CUDAのコンパイラ指定などはCMake引数として渡せる。

```sh
./install.sh -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13
```

インストール後にログアウト・ログインし、Fcitx5の入力メソッドで
**zinai-plus-Hazkey-Fcitx5**を選択。設定でZenzaiを有効化し、利用するデバイスを選ぶ。
GPU番号は環境ごとに異なる。CPU/VulkanでのNative TTT性能は未検証。

既存Hazkeyを置き換える構成。内部のアドオンID・実行ファイル名・保存先は互換性のため`hazkey`を維持し、同時インストールはできない。
同じPREFIXにある置換対象ファイルは退避し、アンインストール時に復元する。
他のPREFIXに入っているHazkeyやディストリビューションのパッケージとの重複は、先に整理する。
パッケージ管理から上書き更新された場合、退避した版は古くなるので注意。

## アンインストール

```sh
./uninstall.sh
# PREFIXを指定して導入した場合は同じ値を指定
PREFIX=/usr/local ./uninstall.sh
```

導入後に変更されたファイルは残す。ユーザーの設定・履歴・モデルは削除しない。

## 保存先

- 設定：`${XDG_CONFIG_HOME:-~/.config}/hazkey`
- モデル・学習状態：`${XDG_DATA_HOME:-~/.local/share}/hazkey`
- 辞書キャッシュ：`${XDG_CACHE_HOME:-~/.cache}/hazkey`

## TTTの検証

4070 SUPERでの検証では更新＋保存が約0.13秒。入力確定から反映は待機込み約3〜4秒。
通常TTTは必ず改善するわけではなく、後続学習による表記の揺れも残る。
[検証結果](distribution/native-ttt-validation.md)と[実装資料](hazkey-server/tools/README-native-ttt.md)を参照。
本配布用インストーラによる新規環境への導入は未検証。

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
