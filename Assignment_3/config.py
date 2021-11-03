from torchvision import transforms
CONFIG = {
    "model_name": {
        "verbose": "Proper Readable Model Name",
        "TRAIN": True,
        "DEVELOPMENT": True,
        "NUM_WORKERS": 8,
        "TRAIN_PARAMS": {
            "BATCH_SIZE": 64,
            "SHUFFLE": True,
            "EPOCHS": 25,
            "LEARNING_RATE": 0.01,
            "MOMENTUM": 0.9,
            "WEIGHT_DECAY": (5e-4),
            "T_MAX": 200,
            "TRANSFORMATIONS": transforms.Compose([
                transforms.Resize(224),
                # transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),
        },
        "VAL_PARAMS": {
            "BATCH_SIZE": 64,
            "SHUFFLE": True,
            "TRANSFORMATIONS": transforms.Compose([
                transforms.Resize(224),
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        },
        "TEST_PARAMS": {
            "BATCH_SIZE": 64,
            "SHUFFLE": True,
            "TRANSFORMATIONS": transforms.Compose([
                transforms.Resize(224),
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        }
    }
}
