import os

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from tqdm import tqdm


def inference(args, model, test_loader, info):
    loop_num = 2 if args["TTA"] else 1

    preds = []
    with torch.no_grad():
        for idx, (images, id_) in enumerate(tqdm(test_loader, total=len(test_loader))):
            for loop_id in range(loop_num):
                images = transforms.RandomHorizontalFlip(p=loop_id)(images)
                images = images.to("cuda")
                if loop_id == 0:
                    pred = model(images)
                else:
                    pred += model(images)
                    pred /= 2
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    if not os.path.exists("submission/" + args["MODEL"]):
        os.makedirs("submission/" + args["MODEL"])

    info["ans"] = preds
    info.to_csv(f"submission/{args['MODEL']}/sub.csv", index=False)
    print(info["ans"].value_counts().sort_index())
    print("Inference Done!")


def infer_logits(
    args, model, train_loader, train_data, valid_loader, valid_data, test_loader, info
):
    if not os.path.exists("submission/" + args["MODEL"]):
        os.makedirs("submission/" + args["MODEL"])

    loop_num = 2 if args["TTA"] else 1

    with torch.no_grad():
        for idx, (images, id_) in enumerate(
            tqdm(train_loader, total=len(train_loader))
        ):
            for loop_id in range(loop_num):
                images = transforms.RandomHorizontalFlip(p=loop_id)(images)
                images = images.to("cuda")
                if loop_id == 0:
                    logit = model(images)
                else:
                    logit += model(images)
                    logit /= 2
            if idx == 0:
                logits = logit.cpu().numpy()
            else:
                logits = np.append(logits, logit.cpu().numpy(), axis=0)

    train_logits = train_data[["img_path"]].copy()
    logits_df = pd.DataFrame(
        logits, columns=[f"l{i:0>2}" for i in range(len(logits[0]))]
    )
    train_logits = pd.concat([train_logits, logits_df], axis=1)
    train_logits.to_csv(
        f"submission/{args['MODEL']}/logits_train_{args['MODEL']}.csv", index=False
    )
    print("Train Logits Done!")

    logits = np.array([])
    with torch.no_grad():
        for idx, (images, id_) in enumerate(
            tqdm(valid_loader, total=len(valid_loader))
        ):
            for loop_id in range(loop_num):
                images = transforms.RandomHorizontalFlip(p=loop_id)(images)
                images = images.to("cuda")
                if loop_id == 0:
                    logit = model(images)
                else:
                    logit += model(images)
                    logit /= 2
            if idx == 0:
                logits = logit.cpu().numpy()
            else:
                logits = np.append(logits, logit.cpu().numpy(), axis=0)

    valid_logits = valid_data[["img_path"]].copy()
    logits_df = pd.DataFrame(
        logits, columns=[f"l{i:0>2}" for i in range(len(logits[0]))]
    )
    valid_logits = pd.concat([valid_logits, logits_df], axis=1)
    valid_logits.to_csv(
        f"submission/{args['MODEL']}/logits_val_{args['MODEL']}.csv", index=False
    )
    print("Validation Logits Done!")

    logits = np.array([])
    with torch.no_grad():
        for idx, (images, id_) in enumerate(tqdm(test_loader, total=len(test_loader))):
            for loop_id in range(loop_num):
                images = transforms.RandomHorizontalFlip(p=loop_id)(images)
                images = images.to("cuda")
                if loop_id == 0:
                    logit = model(images)
                else:
                    logit += model(images)
                    logit /= 2
            if idx == 0:
                logits = logit.cpu().numpy()
            else:
                logits = np.append(logits, logit.cpu().numpy(), axis=0)

    test_logits = info[["ImageID"]].copy()
    logits_df = pd.DataFrame(
        logits, columns=[f"l{i:0>2}" for i in range(len(logits[0]))]
    )
    test_logits = pd.concat([test_logits, logits_df], axis=1)
    test_logits.to_csv(
        f"submission/{args['MODEL']}/logits_test_{args['MODEL']}.csv", index=False
    )
    print("Test Logits Done!")
