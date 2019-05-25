import pandas as pd
import numpy as np

import utils


class GetProcessedData:
    """
    This class is where all preprocessing of data takes place. Excepted what happens in pipeline. Aim is to get feature
    and label array as 2 dimensions np.array.
    """

    # TODO work on dataframe to add features (normalize them ?). see if issue with pipeline cf thoughts logs.
    # Todo randomly shuffle dataset in pandas before creating features?

    def __init__(self, input_fpath, output_fpath):
        self._input_path = input_fpath
        self._output_path = output_fpath

    def _get_df_from_csv(self):
        """
        Compiles input and output training files into a full dataframe.
        :return: pandas.DataFrame
        """
        features_df = pd.read_csv(self._input_path)
        labels_df = pd.read_csv(self._output_path)
        train_df = pd.DataFrame()
        train_df['question'] = features_df['question']
        train_df['intention'] = labels_df['intention']
        self._clean(train_df)
        return train_df

    @staticmethod
    def _clean(train_df):
        """
        Clean duplicates. Remove contradictory ones (same features different labels. Keep first of full duplicates).
        :param train_df: pd.Dataframe
        :return: pd.Dataframe
        """
        contradictory_duplicates = train_df.duplicated(['question'], keep=False) & ~train_df.duplicated(keep=False)
        cleaned_train_df = train_df[~contradictory_duplicates]
        cleaned_train_df = cleaned_train_df.drop_duplicates(subset=['question'], keep="first")
        cleaned_train_df.reset_index(drop=True, inplace=True)
        return cleaned_train_df

    def get_feature_label_arrays(self, show_df=False, show_arrays=False):
        """
        Returns train_data as a pandas.Dataframe and 2 np.arrays of ndim 2: the features and the labels.
        :return: pandas.Dataframe, np.array, np.array
        """
        train_df = self._get_df_from_csv()
        train_df = self._clean(train_df)
        feature = np.array(train_df['question'])
        # if feature.ndim == 1:
        #     feature = feature.reshape(feature.shape[0], 1)
        label = np.array(train_df['intention'])
        # if label.ndim == 1:
        #     label = label.reshape(label.shape[0], 1)
        # TOdo reshape pas bon a cause TFIDF car considere ["bonjour"] au lieu de "bonjour" a voir si reshape utile
        if show_df is True:
            print(train_df.head(10))
            print(train_df.describe(include='all'))
        if show_arrays is True:
            print("Feature shape = {}:\n{}\n\nLabel shape = {}:\n{}\n".format(feature.shape, feature,
                                                                              label.shape, label))
            print(feature.ndim)

        return train_df, feature, label

    def make_xls_from_df(self, fpath):
        train_df = self._get_df_from_csv()
        train_df = self._clean(train_df)
        train_df.to_excel(fpath)


if __name__ == "__main__":
    # GetProcessedData(utils.TRAIN_INPUT_FILE, utils.TRAIN_OUTPUT_FILE).get_feature_label_arrays(show_arrays=True)
    GetProcessedData(utils.TRAIN_INPUT_FILE, utils.TRAIN_OUTPUT_FILE).make_xls_from_df(utils.TRAIN_FILE_XLSX)
