import os
import torch
import model
import config
import numpy as np
import pandas as pd
import albumentations as alb
from wtfml.engine import Engine
from wtfml.data_loaders.image import ClassificationLoader


def test(fold):
    test_img_path  = config.TEST_IMAGES
    device = config.DEVICE
    test_bs = config.TEST_BS

    df_test = pd.read_csv(config.TEST_FILE).reset_index(drop=True)
    model_ = model.SE_Resnext50_32x4d(pretrained=None)
    model_.load_state_dict(torch.load(os.path.join(config.MODEL_PATH, f"model_fold_{fold}.bin")))
    model_.to(device)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    test_aug = alb.Compose([alb.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)])

    test_imgs = df_test.image_name.values.tolist()
    test_imgs = [os.path.join(test_img_path, i + ".jpg") for i in test_imgs]
    test_targets = np.zeros(len(test_imgs))

 
    test_dataset = ClassificationLoader(
        image_paths = test_imgs,
        targets=test_targets,
        resize=(256,256),
        augmentations=test_aug
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size= test_bs, shuffle=False, num_workers=4
    )

    predictions = Engine.predict(test_loader, model_, device=device)
    predictions = np.vstack((predictions)).ravel()

    return predictions


if __name__ == "__main__":

    pred = (3*test(0) + 0.3*test(1) + 0.03*test(2))/3
    sub = pd.read_csv("../input/sample_submission0.csv")
    sub.loc[:, "target"] = pred
    sub.to_csv("submission.csv", index=False)