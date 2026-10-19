#!/bin/bash

CERT="Apple Development: Keita Miwa (29PQN5TTUC)"
XCFRAMEWORK_PATH="build-apple/llama.xcframework"
XCFRAMEWORK_DIR="$(dirname "$XCFRAMEWORK_PATH")"
XCFRAMEWORK_NAME="$(basename "$XCFRAMEWORK_PATH")"
ZIP_OUTPUT="$(pwd)/${XCFRAMEWORK_DIR}/signed-${XCFRAMEWORK_NAME}.zip"

# Remove unnecessary files
echo "🧼 Deleting .DS_Store and AppleDouble files..."
find "$XCFRAMEWORK_PATH" -name '.DS_Store' -delete
find "$XCFRAMEWORK_PATH" -name '._*' -delete

# Find all .framework directories inside the xcframework
find "$XCFRAMEWORK_PATH" -type d -name "llama.framework" | while read -r framework; do
    echo "Signing: $framework"
    codesign --force --timestamp -v --sign "$CERT" "$framework"
done

# xcframework 自体も署名
echo "Signing XCFramework container"
codesign --force --timestamp -v --sign "$CERT" "$XCFRAMEWORK_PATH"

# zip化前に古いzipを削除（ある場合）
if [ -f "$ZIP_OUTPUT" ]; then
    echo "Removing existing zip: $ZIP_OUTPUT"
    rm "$ZIP_OUTPUT"
fi

# XCFramework を zip にアーカイブ
echo "Creating ZIP archive: $ZIP_OUTPUT"
cd "$XCFRAMEWORK_DIR"
export COPYFILE_DISABLE=true
ditto -c -k --keepParent "$XCFRAMEWORK_NAME" "$ZIP_OUTPUT"

# 完了メッセージと checksum 計算
echo "✅ Done! Signed XCFramework archived at: $ZIP_OUTPUT"
swift package compute-checksum "$ZIP_OUTPUT"
