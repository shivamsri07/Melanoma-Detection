import torch
import config
import os
import numpy as np
import pandas as pd
import albumentations as alb
from wtfml.data_loaders.image import ClassificationLoader
from wtfml.utils import EarlyStopping
from wtfml.engine import Engine
from sklearn import metrics


def train(fold):
    train_img_path  = config.TRAINING_IMAGES
    df = pd.read_csv(config.TRAINING_FILE)
    device = config.DEVICE
    epochs = config.EPOCHS
    train_bs = config.TRAIN_BS
    valid_bs = config.VALID_BS

    df_train = df[df.kfold != fold].reset_index(drop = True)
    df_valid = df[df.kfold == fold].reset_index(drop= True)

    model = config.MODEL
    model.to(device)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_aug = alb.Compose([
        alb.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
        alb.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
        alb.Flip(p=0.5)
    ])

    valid_aug = alb.Compose([alb.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)])

    train_imgs = df_train.image_name.values.tolist()
    train_imgs = [os.path.join(train_img_path, i + ".jpg") for i in train_imgs]
    train_targets = df_train.target.values

    valid_imgs = df_valid.image_name.values.tolist()
    valid_imgs = [os.path.join(train_img_path, i + ".jpg") for i in valid_imgs]
    valid_targets = df_valid.target.values

    train_dataset = ClassificationLoader(
        image_paths = train_imgs,
        targets=train_targets,
        resize=(256,256),
        augmentations=train_aug,
    )

    valid_dataset = ClassificationLoader(
        image_paths = valid_imgs,
        targets=valid_targets,
        resize=(256,256),
        augmentations=valid_aug
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size= train_bs, shuffle=False, num_workers=4
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size= valid_bs, shuffle=False, num_workers=4
    )

    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        threshold=0.001,
        mode="max"
    )

    es = EarlyStopping(
        patience=3, mode="max"
    )

    for epoch in range(epochs):
        train_loss = Engine.train(train_loader, model, optimizer, device=device)
        predictions, valid_loss = Engine.evaluate(
            valid_loader, model, device=device
        )
        predictions = np.vstack((predictions)).ravel()
        auc = metrics.roc_auc_score(valid_targets, predictions)
        print(f"Epoch = {epoch}, AUC = {auc}, train_loss={train_loss}, valid_loss={valid_loss}")
        scheduler.step(auc)

        es(auc, model, model_path=f"model_fold_{fold}.bin")
        if es.early_stop:
            print("Early stopping")
            break

if __name__ == "__main__":

    train(0)
    train(1)
    train(2)
    train(3)
    train(4)