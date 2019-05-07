import wget
import zipfile

from imputation_dgm.commandline import create_directories_if_needed
from imputation_dgm.pre_processing.online_news_popularity.transform import main as transform_main


def main():
    create_directories_if_needed("data/online-news-popularity")

    wget.download(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip",
        "data/online-news-popularity/OnlineNewsPopularity.zip"
    )
    print()

    with zipfile.ZipFile("data/online-news-popularity/OnlineNewsPopularity.zip", "r") as zip_file:
        zip_file.extractall("data/online-news-popularity/")

    transform_main(args=[
        "data/online-news-popularity/OnlineNewsPopularity/OnlineNewsPopularity.csv",
        "data/online-news-popularity/features.npy",
        "data/online-news-popularity/labels.npy",
        "data/online-news-popularity/metadata.json"
    ])


if __name__ == "__main__":
    main()
