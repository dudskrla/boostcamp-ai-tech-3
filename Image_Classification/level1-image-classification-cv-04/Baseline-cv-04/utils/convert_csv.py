import pandas as pd

data = pd.read_csv(
    "/opt/ml/boostcamp-ai-tech-3/Image_Classification/level1-image-classification-cv-04/Baseline-cv-04/data/train_info_v1.1.1_ftom6_mtof2.csv"
)
data["gender"] = data["gender"].apply(lambda x: 0 if x == "male" else 1)
data["mask"] = data["mask_type"].apply(
    lambda x: 0 if x == "wear" else (1 if x == "incorrect" else 2)
)
data["age"] = data["age_group"].apply(
    lambda x: 0 if x == "< 30" else (1 if x == ">= 30 and < 60" else 2)
)

data = data.rename(columns={"fname": "img_path", "ans": "label"})
data["img_path"] = data["img_path"].apply(
    lambda x: "/opt/ml/input/data/train/images/" + x
)
data
data.to_csv(
    "/opt/ml/boostcamp-ai-tech-3/Image_Classification/level1-image-classification-cv-04/Baseline-cv-04/data/final_train_df.csv",
    index=False,
    encoding="cp949",
)
