def train_model(model, args):
    """Train the model with the given arguments."""
    # Start training the model
    model.train(
        data=args.data_path,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=640,
        degrees=180,        # 回転の角度
        translate=0.2,      # 平行移動
        scale=0.5,          # スケーリング
        shear=0.1,          # シアー変換
        flipud=0.5,         # 上下反転の確率
        mosaic=0.0,         # Mosaic augmentation の有効化
        cutmix=0.2,
    )

    # Save the trained model
    model.save(args.output)

    return model
