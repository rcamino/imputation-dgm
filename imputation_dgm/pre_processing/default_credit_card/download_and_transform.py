import csv
import wget
import xlrd

from imputation_dgm.commandline import create_directories_if_needed
from imputation_dgm.pre_processing.default_credit_card.transform import main as transform_main


def main():
    create_directories_if_needed("data/default-credit-card")

    wget.download(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default of credit card clients.xls",
        "data/default-credit-card/default_credit_card_clients.xls"
    )
    print()

    workbook = xlrd.open_workbook("data/default-credit-card/default_credit_card_clients.xls")
    sheet = workbook.sheet_by_name("Data")
    with open("data/default-credit-card/default_credit_card_clients.csv", "w") as csv_file:
        writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)

        for row_number in range(1, sheet.nrows):  # ignore first redundant row
            writer.writerow(sheet.row_values(row_number))

    transform_main(args=[
        "data/default-credit-card/default_credit_card_clients.csv",
        "data/default-credit-card/features.npy",
        "data/default-credit-card/labels.npy",
        "data/default-credit-card/metadata.json"
    ])


if __name__ == "__main__":
    main()
