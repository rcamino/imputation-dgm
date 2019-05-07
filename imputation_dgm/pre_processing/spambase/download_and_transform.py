import wget

from imputation_dgm.commandline import create_directories_if_needed
from imputation_dgm.pre_processing.spambase.transform import main as transform_main


def main():
    create_directories_if_needed("data/spambase")

    wget.download(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data",
        "data/spambase/spambase.data"
    )
    print()

    transform_main(args=[
        "data/spambase/spambase.data",
        "data/spambase/features.npy",
        "data/spambase/labels.npy",
        "data/spambase/metadata.json"
    ])


if __name__ == "__main__":
    main()
