# Third-party notices

このフォーク独自の追加・変更部分にはThe Unlicense（`LICENSE`）を適用する。
上流由来のコード・データをThe Unlicenseに再許諾するものではない。

この配布はHazkeyの派生版です。元の著作権表示・ライセンスは削除していません。
変更：Native In-Place TTT、非同期学習・保存、単語登録学習、nico辞書統合、モデル取得先、配布名、導入スクリプト。

| コンポーネント | 出典・ライセンス原文 |
| --- | --- |
| Hazkey | https://github.com/7ka-Hiira/fcitx5-hazkey ・ `LICENSE.upstream-Hazkey` (MIT) |
| 改造llama.cpp / GGML | `hazkey-server/llama.cpp/LICENSE`、各ソース内の表記 |
| 改造AzooKeyKanaKanjiConverter | `hazkey-server/vendor/AzooKeyKanaKanjiConverter/LICENSE` (MIT) |
| azooKey通常辞書 | `hazkey-server/azooKey_dictionary_storage/LICENSE` (Apache-2.0)、同ディレクトリREADME |
| azooKey絵文字辞書 | `hazkey-server/azooKey_emoji_dictionary_storage/README.md` とデータ内の出典表記 |
| dic-nico-intersection-pixiv | https://github.com/ncaq/dic-nico-intersection-pixiv ・ `distribution/licenses/dic-nico-intersection-pixiv-MIT.txt` |

辞書の取得元・取得時点・SHA256は `distribution/dictionaries/*.metadata.json`。
元リポジトリのリビジョンは `distribution/upstream-revisions.json`。改造部分を含むソースをすべて同梱。
Swift Package Managerで取得する依存の著作権表示・ライセンスは取得された各パッケージに含まれる。

モデル重みはこのアーカイブには含まれない。設定画面から取得されるモデル：
https://huggingface.co/soichi1208/zenzai-plus
ベース：https://huggingface.co/Miwa-Keita/zenz-v2.5-small
ライセンス：Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)。
モデルの表記・条件はモデルカードを参照。

外部のビルド・実行依存にはFcitx5（LGPL-2.1-or-later等）、Qt6（対象モジュールはLGPL-3.0等）がある。
これらのライブラリは本ソース配布に同梱していない。バイナリと一緒に再配布する場合は各ライブラリの条件も確認する。
