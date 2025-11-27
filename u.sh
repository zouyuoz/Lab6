#!/bin/bash

# 1. 定義變數
REPO_DIR="$(dirname "$0")" # 取得腳本所在的目錄（即 Lab5 專案根目錄）
COMMIT_MESSAGE=""
DEFAULT_MESSAGE="Auto-update from Colab at $(date +'%Y-%m-%d %H:%M:%S')"

# 2. 進入專案根目錄
cd "$REPO_DIR" || { echo "錯誤：無法進入專案目錄 $REPO_DIR。請檢查腳本位置。"; exit 1; }

echo "--- 正在檢查並更新 Git 倉庫 ($PWD) ---"

# 3. 檢查是否有本地改動
# 檢查 git status 的輸出是否包含 "nothing to commit"
if git status --porcelain | grep -q .; then
    echo "發現本地未提交的改動..."
else
    echo "沒有發現未提交的改動，腳本結束。"
    exit 0
fi

# 4. 處理提交訊息 very good very bad
if [ -n "$1" ]; then
    COMMIT_MESSAGE="$1"
else
    COMMIT_MESSAGE="$DEFAULT_MESSAGE"
fi

# 5. Git 操作
# a. 將所有更改加入暫存區 (包括新增、修改、刪除)
echo "1. 正在暫存所有檔案 (git add .)..."
git add .

# b. 提交更改
echo "2. 正在提交更改 (git commit -m \"$COMMIT_MESSAGE\")..."
if git commit -m "$COMMIT_MESSAGE"; then
    # c. 推送到遠端倉庫
    echo "3. 正在推送到遠端倉庫 (git push)..."
    if git push; then
        echo "✅ 成功將專案更新並推送到 GitHub！"
    else
        echo "❌ 錯誤：推送到 GitHub 失敗。請檢查您的連線或權限 (例如：是否使用 Personal Access Token)。"
    fi
else
    echo "❌ 錯誤：Git 提交 (Commit) 失敗。請檢查您的本地 Git 配置。"
fi

# 6. 返回腳本執行前的目錄 (可選，但在 Colab 中不執行也無妨)
# cd - > /dev/null