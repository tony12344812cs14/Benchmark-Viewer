# Benchmark-Viewer

MMLU 題目、正確答案與各主流模型答對率的瀏覽器。

**線上網頁：** https://tony12344812cs14.github.io/Benchmark-Viewer/

## 功能
- 題目與 A–D 選項，正解標綠，顯示每個選項有幾個模型選
- 本題正確答案與答對模型數 / 答對率
- 各模型本題選擇、對錯，以及在目前篩選下的整體答對率
- 題目總覽地圖（依答對比例上色）、模型答對率總覽、科目 × 模型熱圖
- 依科目 / 難度 / 關鍵字篩選；← → 切換題目，R 隨機

## 使用真實資料（RouterBench）
網頁預設顯示**示範資料**（自編題目、亂數產生的對錯，非真實結果）。換成真實資料：

```bash
# 1. 從 HuggingFace 下載 withmartian/routerbench 的 routerbench_0shot.pkl
# 2. 轉成網頁用的 JSON
python routerbench_to_json.py routerbench_0shot.pkl -o mmlu_routerbench.json
# 3. 打開網頁，按「載入 JSON」或把檔案拖進頁面
```

RouterBench 沒有正解欄位，正確答案是用「答對模型所選字母」的多數決反推；沒有模型答對的題目會顯示「無法反推」。

JSON 格式：
```json
{"models": ["gpt-4-1106-preview", "..."],
 "items": [{"id": "...", "subject": "virology", "prompt": "...", "gold": "B",
            "results": {"gpt-4-1106-preview": 1}, "picks": {"gpt-4-1106-preview": "B"}}]}
```
