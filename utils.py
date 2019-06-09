import pandas as pd

TRAIN_INPUT_FILE = 'files/input_train.csv'
TRAIN_OUTPUT_FILE = 'files/output_train.csv'
TRAIN_FILE_XLSX = 'files/train.xlsx'


def print_grid_search_results(df, filename):
    output_path = filename + '.xlsx'
    writer = pd.ExcelWriter(output_path)
    df.to_excel(writer)
    writer.save()
