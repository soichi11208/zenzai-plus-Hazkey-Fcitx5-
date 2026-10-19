# Gen

azooKeyのための絵文字辞書データのレポジトリです。

## Environment Setup

```bash
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## 絵文字データの更新

```bash
python3 scripts/emoji.py
```

## 新しいUnicodeバージョンへの対応作業

1. 現状のコードのまま実行し、動作確認

```bash
python3 scripts/emoji.py
```

2. 差分がないことを確認
```bash
git diff
```

3. `data/`のファイルを更新

```bash
curl https://raw.githubusercontent.com/google/mozc/refs/heads/master/src/data/emoji/emoji_data.tsv > data/emoji_data.tsv
curl https://unicode.org/Public/emoji/latest/emoji-sequences.txt > data/emoji-sequences.txt
curl https://unicode.org/Public/emoji/latest/emoji-zwj-sequences.txt > data/emoji-zwj-sequences.txt
curl https://unicode.org/Public/emoji/latest/emoji-test.txt > data/emoji-test.txt
curl https://raw.githubusercontent.com/unicode-org/cldr/refs/heads/main/common/annotations/ja.xml > data/ja.xml
curl https://raw.githubusercontent.com/unicode-org/cldr/refs/heads/main/common/annotationsDerived/ja.xml > data/ja_derived.xml
```

4. `scripts/emoji.py`の実行部分で`version_targets`の指定を更新
5. 再実行

```bash
python3 scripts/emoji.py
```
6. 差分を確認
```bash
# 例
diff './EmojiDictionary/emoji_all_E15.1.txt' './EmojiDictionary/emoji_all_E16.0.txt'
57a58
> 🫩️     かお,くま,つかれた,てつや,ねむい,めにくまがあるかお,徹夜,疲れた,目にくまがある顔,眠い,顔
565a567
> 🫆️     かんしき,しもん,せいたいにんしょう,せきゅりてぃ,指紋,生体認証,鑑識
727a730
> 🪾️     かれき,かんばつ,はのないき,ふもう,ふゆ,不毛,冬,干ばつ,枯れ木,葉のない木
765a769
> 🫜️     かぶ,こんさい,にわ,ね,びーつ,やさい,庭,根,根菜,野菜
1236a1241
> 🪉️     あい,おんがく,おーけすとら,がっき,きゅーぴっど,げんがっき,はーぷ,弦楽器,愛,楽器,音楽
1374a1380
> 🪏️     あな,しゃべる,しょべる,すき,すこっぷ,ほる,掘る,穴,鋤
1559a1566
> 🫟️     しぶき,すぷらっしゅ,とびちったぺんき,とびちり,ぺんき,飛び散ったぺんき,飛び散り,飛沫
```
7. 必要に応じてバグを修正/読みなどを追加