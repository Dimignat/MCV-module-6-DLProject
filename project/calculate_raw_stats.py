import pandas as pd

from .utils import INV_LABEL_MAP, load_data


def calculate_raw_stats() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_images, train_labels, test_images = load_data()

    train_df_data, test_df_data = [], []
    for img, label in zip(train_images, train_labels):
        train_df_data.append(
            {
                "img_size": img.size,
                "img_area": img.size[0] * img.size[1],
                "klass": INV_LABEL_MAP[label[0]],
                "bbox_area": label[-1] * label[-2],
                "subset": "train",
            }
        )
    for img in test_images:
        test_df_data.append(
            {
                "img_size": img.size,
                "img_area": img.size[0] * img.size[1],
                "klass": None,
                "bbox_area": None,
                "subset": "test",
            }
        )

    return pd.DataFrame(train_df_data), pd.DataFrame(test_df_data)


def main():
    train_df, test_df = calculate_raw_stats()

    train_df = train_df.sort_values("img_area")
    min_sz, max_sz = (
        train_df.img_size.iloc[0],
        train_df.img_size.iloc[len(train_df) - 1],
    )
    median_sz = train_df.img_size.iloc[len(train_df) // 2]
    print("TRAIN:")
    print("count:", len(train_df))
    print("image size (min, median, max):", min_sz, median_sz, max_sz)
    print(
        "bbox relative area (min, median, max):",
        *train_df.bbox_area.quantile([0, 0.5, 1]).tolist()
    )
    print("class count:", train_df.klass.value_counts().to_dict())
    print("\n")

    test_df = test_df.sort_values("img_area")
    min_sz, max_sz = test_df.img_size.iloc[0], test_df.img_size.iloc[len(test_df) - 1]
    median_sz = test_df.img_size.iloc[len(test_df) // 2]
    print("TEST:")
    print("count:", len(test_df))
    print("image size (min, median, max):", min_sz, median_sz, max_sz)


if __name__ == "__main__":
    main()
