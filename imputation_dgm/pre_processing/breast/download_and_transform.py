import wget

from imputation_dgm.commandline import create_directories_if_needed
from imputation_dgm.pre_processing.breast.transform import main as transform_main


def main():
    create_directories_if_needed("data/breast")

    wget.download(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
        "data/breast/wdbc.data"
    )
    print()

    transform_main(args=[
        "data/breast/wdbc.data",
        "data/breast/features.npy",
        "data/breast/labels.npy",
        "data/breast/metadata.json"
    ])


if __name__ == "__main__":
    main()
