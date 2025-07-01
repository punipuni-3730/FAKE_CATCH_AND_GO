import cv2
import numpy as np
import json
import asyncio
from yolov5 import YOLOv5
import torch
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# グローバル変数
price_db = {}
PRICE_FILE = "prices.json"
model = None  # モデルを遅延ロード

def load_prices():
    """価格データをロード"""
    global price_db
    try:
        with open(PRICE_FILE, 'r') as f:
            price_db = json.load(f)
    except:
        price_db = {}
    print(f"ロードされた価格データ: {price_db}")

def save_prices():
    """価格データを保存"""
    with open(PRICE_FILE, 'w') as f:
        json.dump(price_db, f, indent=4)

def init_model(device):
    """YOLOv5モデルを初期化"""
    global model
    if model is None:
        print("モデルをロード中...")
        model = YOLOv5('yolov5n.pt', device=device)  # 軽量モデル
        print(f"モデルデバイス: {next(model.model.parameters()).device}")

def detect_objects(frame):
    """YOLOv5で物体を検出（1アイテム1つのみ）"""
    frame = cv2.resize(frame, (640, 640))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(frame)
    detections = results.pred[0].cpu().numpy()
    objects = []
    print(f"検出数: {len(detections)}")
    for det in detections:
        class_id = int(det[5])
        label = results.names[class_id]
        confidence = det[4]
        print(f"物体: {label}, 信頼度: {confidence:.2f}")
        if confidence > 0.5 and label not in objects:  # 1アイテム1つのみ
            objects.append(label)
    return objects

async def price_setting_mode(cap):
    """価格設定モード"""
    print("価格設定モード: 物体をカメラに映し、名前と価格を入力してください。終了するには 'q' または 'Esc' を押してください。")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("カメラの読み込みに失敗しました")
            break

        objects = detect_objects(frame)
        for obj in objects:
            if obj not in price_db:
                price = input(f"物体 '{obj}' の価格を入力してください（円）: ")
                try:
                    price_db[obj] = float(price)
                    print(f"{obj} の価格を {price} 円に設定しました")
                except ValueError:
                    print("無効な価格です。数値を入力してください。")

        cv2.imshow('Price Setting Mode', frame)
        key = cv2.waitKey(30) & 0xFF
        if key in [ord('q'), 27]:
            print("qまたはEscが検出され、価格設定モードを終了します")
            break

        await asyncio.sleep(0.01)

    save_prices()
    return True  # モード選択に戻る

async def production_mode(cap):
    """本番モード: 開始時にアイテムを記録、qで精算"""
    print("本番モード: 's'キーでアイテム検出を開始、'q'または'Esc'で精算または終了します。")
    initial_objects = None
    total_price = 0.0
    started = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("カメラの読み込みに失敗しました")
            break

        if started:
            # 検出中のアイテムを画面に表示
            if initial_objects:
                for obj in initial_objects:
                    if obj in price_db:
                        cv2.putText(frame, f"検出中: {obj} ({price_db[obj]:.2f} yen)", 
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Production Mode', frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('s') and not started:
            # 's'キーで初期アイテムを検出
            print("アイテム検出を開始します...")
            initial_objects = detect_objects(frame)
            if initial_objects:
                print(f"検出されたアイテム: {initial_objects}")
                started = True
            else:
                print("アイテムが検出されませんでした。もう一度's'を押してください。")

        elif key == ord('q') and started:
            # 'q'キーで現在のアイテムを取得し、精算
            current_objects = detect_objects(frame)
            print(f"現在のアイテム: {current_objects}")
            for obj in initial_objects:
                if obj not in current_objects and obj in price_db:
                    total_price += price_db[obj]
                    print(f"物体 '{obj}' が購入されました。価格: {price_db[obj]} 円")
            if total_price > 0:
                print(f"精算処理: 合計金額 {total_price:.2f} 円")
            else:
                print("購入された商品はありません")
            started = False  # 精算後、リセット
            initial_objects = None

        elif key in [ord('q'), 27]:
            # 'q'または'Esc'でモード選択に戻る
            print("qまたはEscが検出され、本番モードを終了します")
            if total_price > 0:
                print(f"最終精算: 合計金額 {total_price:.2f} 円")
            break

        await asyncio.sleep(0.01)

    return True  # モード選択に戻る

async def main():
    """メインループ"""
    device = 'cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu'
    print(f"使用デバイス: {device}")
    if device.startswith('cuda'):
        try:
            print(f"GPU名: {torch.cuda.get_device_name(0)}")
        except:
            print("CUDAデバイスが見つかりません。CPUにフォールバックします。")
            device = 'cpu'

    load_prices()

    cap = None
    while True:
        if cap is None:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not cap.isOpened():
                print("Webカメラを起動できません")
                return

        init_model(device)  # モデルをモード実行時にロード

        mode = input("モードを選択してください (1: 価格設定モード, 2: 本番モード, q: 終了): ")
        if mode == '1':
            if await price_setting_mode(cap):
                continue  # モード選択に戻る
        elif mode == '2':
            if await production_mode(cap):
                continue  # モード選択に戻る
        elif mode == 'q':
            print("プログラムを終了します")
            break
        else:
            print("無効なモードです。1, 2, または q を入力してください。")

        cap.release()
        cv2.destroyAllWindows()

cap = None  # グローバルスコープでcapを定義

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("プログラムがCtrl+Cで終了しました")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()