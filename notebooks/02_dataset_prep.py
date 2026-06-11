import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_IMAGES       = Path("data/raw/img_align_celeba/img_align_celeba")
ATTR_FILE        = Path("data/raw/list_attr_celeba.csv")
OUTPUT_DIR       = Path("data/processed")
SAMPLES_PER_CLASS = 3000


def load_attributes():
    df = pd.read_csv(ATTR_FILE, header=0, index_col="image_id")
    df = df.replace(-1, 0)
    return df


def classify_row(row) -> str | None:
    hm  = row["Heavy_Makeup"]
    lip = row["Wearing_Lipstick"]

    if hm == 0 and lip == 0:
        return "no_makeup"
    if hm == 0 and lip == 1:
        return "natural"
    if hm == 1:
        return "heavy"
    return None


def build_dataset():
    df = load_attributes()
    df["class"] = df.apply(classify_row, axis=1)
    df = df.dropna(subset=["class"])

    print("Distribuição antes do balanceamento:")
    print(df["class"].value_counts())

    for cls in df["class"].unique():
        subset = df[df["class"] == cls].sample(
            min(SAMPLES_PER_CLASS, len(df[df["class"] == cls])),
            random_state=42
        )
        train, temp = train_test_split(subset, test_size=0.3, random_state=42)
        val, test   = train_test_split(temp,   test_size=0.5, random_state=42)

        for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
            dest = OUTPUT_DIR / split_name / cls
            dest.mkdir(parents=True, exist_ok=True)
            for img_name in split_df.index:
                src = RAW_IMAGES / img_name
                if src.exists():
                    shutil.copy(src, dest / img_name)

    print("\n✅ Dataset organizado:")
    for split in ["train", "val", "test"]:
        for cls in df["class"].unique():
            n = len(list((OUTPUT_DIR / split / cls).glob("*.jpg")))
            print(f"  {split}/{cls}: {n} imagens")


if __name__ == "__main__":
    build_dataset()