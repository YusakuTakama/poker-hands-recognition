import streamlit as st
from PIL import Image
from app import HandsRecognizer
from app.hands_recognition import HandRank
from detection_model import YoLoViTDetector
from collections import Counter


# HandRankの英語名と日本語名の対応マッピング
HAND_RANK_JAPANESE = {
    'HIGH_CARD': 'ハイカード',
    'ONE_PAIR': 'ワンペア',
    'TWO_PAIR': 'ツーペア',
    'THREE_OF_A_KIND': 'スリーカード',
    'STRAIGHT': 'ストレート',
    'FLUSH': 'フラッシュ',
    'FULL_HOUSE': 'フルハウス',
    'FOUR_OF_A_KIND': 'フォーカード',
    'STRAIGHT_FLUSH': 'ストレートフラッシュ',
    'ROYAL_FLUSH': 'ロイヤルストレートフラッシュ'
}


def upload_image():
    """画像アップロード用UI"""
    uploaded_file = st.file_uploader("ポーカーハンドの画像をアップロードしてください。役の判定を行い、記録します。", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # アップロードされたファイルをPIL.Imageオブジェクトに変換
        image = Image.open(uploaded_file)
        st.image(image, caption='アップロードされた画像', use_container_width=True)
        return image
    return None


def display_hand_statistics():
    """役の統計情報を表示する関数"""
    if 'results' not in st.session_state or not st.session_state.results:
        return

    # 役の集計を行う
    hand_ranks = [res['hand_rank'] for res in st.session_state.results if 'hand_rank' in res]
    rank_counter = Counter(hand_ranks)

    st.subheader("役の統計情報")

    # 全ての役を表示する（出現回数0の役も含む）
    all_ranks = [rank.name for rank in HandRank]
    for rank_name in all_ranks:
        count = rank_counter.get(rank_name, 0)
        japanese_name = HAND_RANK_JAPANESE.get(rank_name, rank_name)
        st.write(f"{japanese_name}：{count}回")

    # 総対戦数
    st.write(f"総計：{len(hand_ranks)}回")


def show():

    # Streamlitのセッション状態で結果を保持
    if 'results' not in st.session_state:
        st.session_state.results = []

    st.title("ポカログNet - ポーカーハンド履歴")
    if 'model' not in st.session_state:
        st.session_state.model = YoLoViTDetector()

    image = upload_image()  # PIL Imageオブジェクト

    if image:
        if st.button("カード認識処理を開始"):
            current_result = {}  # 現在の処理結果を初期化

            # 1. BBox検出 (入力はPIL Image)．
            cards = st.session_state.model.detect(image)
            if cards is None:
                # エラーを吐いて中断
                st.error("カードの検出に失敗しました。画像を確認してください。")
            current_result['cards'] = cards

            # 2. カード分類と役判定を行う
            recognizer = HandsRecognizer(cards)
            hand_rank, best_hand = recognizer.evaluate()
            current_result['hand_rank'] = hand_rank.name
            current_result['best_hand'] = best_hand

            # 3. 役判定結果を記録
            st.session_state.results.append(current_result)
            st.success("カード認識と役判定が完了しました。結果を保存しました。")
            japanese_name = HAND_RANK_JAPANESE.get(hand_rank.name, hand_rank.name)
            st.write(f" {japanese_name}")
            st.write("ハンド:  " + ", ".join(card.omittion() for card in best_hand))

    # 保存された結果の表示
    if st.session_state.results:
        st.subheader("これまでの役履歴")
        for i, res in enumerate(st.session_state.results):
            st.write(f"--- 結果 {i+1} ---")
            if 'hand_rank' in res:
                japanese_name = HAND_RANK_JAPANESE.get(res['hand_rank'], res['hand_rank'])
                st.write(f"{japanese_name}")
            if 'best_hand' in res:
                st.write("ハンド:  " + ", ".join(card.omittion() for card in res['best_hand']))
            st.write("--------------------")

        # 役の統計情報を表示
        display_hand_statistics()


if __name__ == "__main__":
    show()
