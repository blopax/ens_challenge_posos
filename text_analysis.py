import re
import unidecode
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TextAnalysis(BaseEstimator, TransformerMixin):
    """
    Class extracting more features.
    """

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        df = pd.Series(x)
        length = len(df)
        time_check = df.apply(lambda x: self.check_time(x))
        # print(time_check)
        time_check = np.array(time_check)
        return time_check.reshape(length, 1)
        # feature = np.array(train_df['question']) #, ndmin=2)
        # label = np.array(train_df['intention'])
        # return self

    @staticmethod
    def check_time(text):
        """
        Check if words related to time
        :param text:
        :return:
        """

        text = text.lower()
        text = unidecode.unidecode(text)
        time_words = r"""[ 0-9]((an|heure|semaine|jour|seconde)s?|mois|mins?|s|sec|h)([0-9 \.]|$)"""
        time_words += r"""|depuis|pendant|hier|demain|matin|midi|soir|nuit"""
        time_words += r"""|janvier|fevrier|mars|avril|mai|juin|juillet|aout|septembre|octobre|novembre|decembre"""
        time_words += r"""|lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche"""

        boolean = re.search(time_words, text) is not None
        return int(boolean)


if __name__ == "__main__":
    t = "J'ai 3ans "
    print(check_time(t))
    t = "J'ai ansi"
    print(check_time(t))

    t = "J'aime mon moi moisi "
    print(check_time(t))
