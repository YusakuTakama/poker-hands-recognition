# Poker Hands Recognition

## 概要
ポーカーハンド認識プロジェクトでは、Ultralytics YOLO を用いたトランプカードの検出から、カード分類モデルによるスートと数字の認識、ハンドの評価までを一貫して実行できるワークフローを提供します。

主な機能:
- カード検出モデルのトレーニングおよび検証 (detection_model フォルダ)
- 検出結果の画像への描画と保存 (show_result.py)
- カード分類とハンド評価を行うストリームリットアプリ (pages/poker_log_net.py)
- 単一カード分類用モデルのトレーニング/推論 (classification_model フォルダ)
- サンプルデータセットフォルダ作成スクリプト (mk_cards.py)

## 目次
1. [環境構築](#環境構築)
2. [フォルダ構成](#フォルダ構成)
3. [データ準備](#データ準備)
4. [検出モデルのトレーニング](#検出モデルのトレーニング)
5. [検出モデルのバリデーション](#検出モデルのバリデーション)
6. [検出結果の可視化](#検出結果の可視化)
7. [ストリームリットアプリ (Poker Log Net)](#ストリームリットアプリ-poker-log-net)
8. [分類モデル](#分類モデル)
9. [サンプルデータ作成](#サンプルデータ作成)
10. [ライセンス](#ライセンス)

---

## 環境構築
1. リポジトリをクローン:
   ```bash
   git clone <このリポジトリのURL>
   cd poker-hands-recognition
   ```
2. Python 環境作成 (推奨: venv または conda):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. 依存パッケージのインストール:
   ```bash
   pip install -r requirements.txt
   ```

## フォルダ構成
```
├── checkpoints/                # 検出モデル重み保存先
├── detection_model/           # 検出モデル関連コード
│   ├── model_factory.py       # YOLOモデル構築
│   ├── train.py               # トレーニング関数
│   ├── val.py                 # バリデーション実装
│   └── main.py                # CLI エントリポイント
├── model/                     # (旧)検出モデルディレクトリ
├── classification_model/      # カード分類モデル関連
├── pages/                     # ストリームリットアプリ (Poker Log Net)
├── mk_cards.py                # サンプル画像フォルダ作成スクリプト
├── show_result.py             # 検出結果を描画 & 保存
├── main.py                    # ストリームリットアプリ起動
├── requirements.txt           # 必要パッケージ一覧
├── README.md                  # このファイル
└── その他スクリプト・データ
```

## データ準備
- 検出用データセット: `detection_model/data/` 以下に `images/` と `labels/` を用意し、Ultralytics セット形式の `data.yaml` を配置
- カード分類用データセット: `dataset/single_card/train` と `dataset/single_card/val` にクラスごとに画像を配置

## 検出モデルのトレーニング
```bash
python detection_model/main.py \
  --data_path detection_model/data/data.yaml \
  --batch_size 16 \
  --epochs 100 \
  --output detection_model/checkpoints \
  --model yolov11
```

## 検出モデルのバリデーション
```bash
python detection_model/val.py \
  --weights detection_model/checkpoints/best.pt \
  --data-path detection_model/data/data.yaml \
  --batch-size 32
```

## 検出結果の可視化
トレーニング済み重みとテスト画像を指定して実行:
```bash
python show_result.py
```
`show_result.py` 上部の `WEIGHTS_PATH` と `IMAGE_PATH` を環境に合わせて編集してください。

## ストリームリットアプリ (Poker Log Net)
```bash
streamlit run main.py
```
- 画像アップロード後、「カード認識処理を開始」をクリックして履歴表示、役統計を確認できます。

## 分類モデル
`classification_model` フォルダ内に、単一カード分類用のモデルチェックポイントが格納されています。詳細は内部の README またはスクリプトを参照してください。

## サンプルデータ作成
`mk_cards.py` を実行すると `dataset/cards/val/<クラス名>/` にサンプル画像をコピーします。
```bash
python mk_cards.py
```

## ライセンス
MIT License
