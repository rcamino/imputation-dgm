import wget

from imputation_dgm.commandline import create_directories_if_needed
from imputation_dgm.pre_processing.letter_recognition.transform import main as transform_main


def main():
    create_directories_if_needed("data/letter-recognition")

    wget.download(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data",
        "data/letter-recognition/letter-recognition.data"
    )
    print()

    transform_main(args=[
        "data/letter-recognition/letter-recognition.data",
        "data/letter-recognition/features.npy",
        "data/letter-recognition/labels.npy",
        "data/letter-recognition/metadata.json"
    ])


if __name__ == "__main__":
    main()
