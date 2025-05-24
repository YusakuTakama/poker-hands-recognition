import streamlit as st
from PIL import Image
from detection_model.model_factory import build_model
from app import HandsRecognizer


# Streamlitのセッション状態で結果を保持
if 'results' not in st.session_state:
    st.session_state.results = []


def upload_image():
    """画像アップロード用UI"""
    uploaded_file = st.file_uploader("ポーカーハンドの画像をアップロードしてください。役の判定を行い、記録します。", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # アップロードされたファイルをPIL.Imageオブジェクトに変換
        image = Image.open(uploaded_file)
        st.image(image, caption='アップロードされた画像', use_column_width=True)
        return image
    return None


def main():
    st.title("ポカログNet - ポーカーハンド履歴")
    if 'model' not in st.session_state:
        st.session_state.model = build_model("yolov11")

    image = upload_image()  # PIL Imageオブジェクト

    if image:
        if st.button("カード認識処理を開始"):
            current_result = {}  # 現在の処理結果を初期化

            # 1. BBox検出 (入力はPIL Image)．7つになるまで繰り返す．
            while True:
                bboxes = st.session_state.model.detect(image)
                if len(bboxes) == 7:
                    break
                else:
                    st.warning(f"検出されたカードの数が7枚未満です。再度検出を試みます... (現在の枚数: {len(bboxes)})")

            cards_bbox = st.session_state.model.detect(image)
            current_result['bboxes'] = cards_bbox

            # 2. カード分類と役判定を行う
            recognizer = HandsRecognizer(cards_bbox)
            hand_rank, best_hand = recognizer.evaluate()
            current_result['hand_rank'] = hand_rank.value
            current_result['best_hand'] = [str(card) for card in best_hand]

            # 3. 役判定結果を記録
            st.session_state.results.append(current_result)
            st.success("カード認識と役判定が完了しました。結果を保存しました。")
            st.write(f"役: {hand_rank.name}")
            st.write("ベストハンド:")
            for card_str in best_hand:
                st.write(f"  {card_str}")

    # 保存された結果の表示
    if st.session_state.results:
        st.subheader("保存された認識結果")
        for i, res in enumerate(st.session_state.results):
            st.write(f"--- 結果 {i+1} ---")
            if 'classified_cards' in res and res['classified_cards']:
                st.write("分類されたカード:")
                # res['classified_cards'] が Card オブジェクトのリストの場合
                for card_obj in res['classified_cards']:
                    st.write(f"  {card_obj}")  # Cardオブジェクトが__str__を持つ前提
            elif 'bboxes' in res:
                st.write(f"検出されたBBoxの数: {len(res['bboxes'])}")

            if 'hand_rank' in res:
                st.write(f"役: {res['hand_rank']}")
            if 'best_hand' in res:
                st.write("ベストハンド:")
                for card_str in res['best_hand']:
                    st.write(f"  {card_str}")
            st.write("--------------------")


if __name__ == "__main__":
    main()
