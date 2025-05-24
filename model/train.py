def train_model(model, args):

    # Start training the model
    model.train(data=args.data_path, epochs=args.epochs, batch=args.batch_size)

    # Save the trained model
    model.save(args.output)

    return model
