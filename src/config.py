import model

TRAINING_IMAGES = "../input/train"
TRAINING_FILE = "../input/train_folds.csv"
TEST_IMAGES = "../input/test"
TEST_FILE = "../input/test.csv"
DEVICE = "cuda"
EPOCHS = 7
TRAIN_BS = 16
VALID_BS = 8
TEST_BS = 8
MODEL = model.SE_Resnext50_32x4d(pretrained="imagenet")
MODEL_PATH = "."
