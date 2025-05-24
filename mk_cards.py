import os
import shutil

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
parent_folder = "dataset/cards"
subfolder = "val"

sub_path = os.path.join(parent_folder, subfolder)
os.makedirs(sub_path, exist_ok=True)

# 各クラスフォルダを作成
for cls in classes:
    class_path = os.path.join(sub_path, cls)
    os.makedirs(class_path, exist_ok=True)


source_img_folder = "dataset/cards/images/val/jpg"
sample_size = 1

# 各クラスフォルダにサンプル画像をコピー
# source_img_folder内のjpgファイルをソートしてから，上からsample_size枚ずつ，classesの順にクラスフォルダにコピー


# 画像ファイル一覧を取得してソート
all_images = sorted([
    f for f in os.listdir(source_img_folder)
    if f.lower().endswith(".jpg")
])

# エラー防止：画像数が足りない場合の確認
total_required = sample_size * len(classes)
if len(all_images) < total_required:
    raise ValueError(f"画像が不足しています。必要: {total_required} 枚, 実際: {len(all_images)} 枚")

# 画像を各クラスフォルダにコピー
index = 0
for cls in classes:
    class_dir = os.path.join(sub_path, cls)
    for i in range(sample_size):
        src_img_path = os.path.join(source_img_folder, all_images[index])
        dst_img_path = os.path.join(class_dir, all_images[index])
        shutil.copy(src_img_path, dst_img_path)
        index += 1

print("サンプル画像のコピーが完了しました。")
