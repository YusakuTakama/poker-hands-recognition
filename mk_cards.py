import os

# クラス名リスト（トランプのカード）
classes = [
    "1h", "1d", "1c", "1s",
    "2h", "2d", "2c", "2s",
    "3h", "3d", "3c", "3s",
    "4h", "4d", "4c", "4s",
    "5h", "5d", "5c", "5s",
    "6h", "6d", "6c", "6s",
    "7h", "7d", "7c", "7s",
    "8h", "8d", "8c", "8s",
    "9h", "9d", "9c", "9s",
    "th", "td", "tc", "ts",
    "jh", "jd", "jc", "js",
    "qh", "qd", "qc", "qs",
    "kh", "kd", "kc", "ks"
]

# 親フォルダ
parent_folder = "cards"

# train と val フォルダの作成
subfolders = ["train", "val"]
for sub in subfolders:
    sub_path = os.path.join(parent_folder, sub)
    os.makedirs(sub_path, exist_ok=True)

    # 各クラスフォルダを作成
    for cls in classes:
        class_path = os.path.join(sub_path, cls)
        os.makedirs(class_path, exist_ok=True)
