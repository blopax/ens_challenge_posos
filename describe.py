import pandas as pd
import copy


TRAIN_INPUT_FILE = 'files/input_train.csv'
TRAIN_OUTPUT_FILE = 'files/output_train.csv'


if __name__ == "__main__":
    try:
        train_input_data = pd.read_csv(TRAIN_INPUT_FILE)
        train_output_data = pd.read_csv(TRAIN_OUTPUT_FILE)
        train_data = copy.deepcopy(train_input_data)
        train_data['intention'] =opyain_output_data['intention']
        print(train_data.describe(include='all'))

    except FileNotFoundError as err:
        print(err)
