import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
# from sklearn.ensemble import VotingClassifier

from sklearn import naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

import utils
from data_preprocessing import GetProcessedData
from score_utils import ScoreUtils
from clf_parameters_setter import ClfParamsSetter


class ClassifierTrainer:
    """
    Main training class. Enable to launch training with different options.
    """

    def __init__(self, input_fpath, output_fpath):
        self._input_path = input_fpath
        self._output_path = output_fpath

    def split_data(self):
        train_df, x, y = GetProcessedData(self._input_path, self._output_path).get_feature_label_arrays()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    @staticmethod
    def set_pipeline(parameters):
        pipeline_dic = {}
        for k, v in parameters.items():
            pipeline_dic[k] = Pipeline(memory=None, steps=[('vect', v['vect'][0]), ('feat_sel', v['feat_sel'][0]),
                                                           ('clf', v['clf'][0])])
        return pipeline_dic

    def unique_clf(self):
        x_train, x_test, y_train, y_test = self.split_data()
        pipeline = Pipeline(memory=None, steps=[
            ('vect', (TfidfVectorizer(use_idf=True, smooth_idf=True, ngram_range=(1, 3)))),
            ('feat_sel', SelectKBest(chi2, k=2000)),
            # ('clf', (dummy.DummyClassifier(strategy="stratified", random_state=0)))
            ('clf', naive_bayes.BernoulliNB(alpha=0.0001))
        ])

        clf = pipeline
        clf.fit(x_train, y_train)
        predicted_train = clf.predict(x_train)
        predicted_test = clf.predict(x_test)

        train_metrics_dic, test_metrics_dic = ScoreUtils.metrics_getter(y_train, y_test, predicted_train,
                                                                        predicted_test)
        print("Accuracy score on train: {}".format(train_metrics_dic['acc_score']))
        print("Accuracy score on test: {}".format(test_metrics_dic['acc_score']))

    def search_cv(self):
        x_train, x_test, y_train, y_test = self.split_data()
        print(x_train, y_train, len(x_train), len(y_train), len(y_test))
        dict_parameters, parameters_list, dict_clf, n_iter = ClfParamsSetter('search').get_parameters()

        pipleine_dic = self.set_pipeline(dict_parameters)
        skf = StratifiedKFold(n_splits=4, shuffle=False)  # warning si ds y il existe des classes ac - de n_splits
        scorer_dic, refit = ScoreUtils.get_scorers_info()

        clf = GridSearchCV(pipleine_dic[list(pipleine_dic.keys())[0]], parameters_list, cv=skf, n_jobs=1,
                           scoring=scorer_dic, refit=refit, return_train_score=False)
        # clf = RandomizedSearchCV(pipleine_dic[list(pipleine_dic.keys())[0]], dict_clf,
        #                          n_iter=n_iter, cv=skf, n_jobs=1,
        #                       scoring=scorer_dic, refit=refit, return_train_score=False)
        clf.fit(x_train, y_train)
        predicted_train = clf.predict(x_train)
        predicted_test = clf.predict(x_test)

        df_verify = pd.DataFrame()
        df_verify['truth'] = y_test
        df_verify['predicted'] = predicted_test
        print(df_verify[df_verify['predicted'] != df_verify['truth']])
        print(y_test, predicted_test)
        train_metrics_dic, test_metrics_dic = ScoreUtils.metrics_getter(y_train, y_test, predicted_train,
                                                                        predicted_test)
        print("Accuracy score on train: {}".format(train_metrics_dic['acc_score']))
        print("Accuracy score on test: {}".format(test_metrics_dic['acc_score']))

    def random_cv(self):
        x_train, x_test, y_train, y_test = self.split_data()
        dict_parameters, parameters_list, parameters_distrib, n_iter = ClfParamsSetter('random').get_parameters()
        print("Dict paramet = {} \n\n, parameters_list = {}\n\n, parameters_dist = {} \n\nn_iter = {}".format(
            dict_parameters, parameters_list, parameters_distrib, n_iter))
        pipleine_dic = self.set_pipeline(dict_parameters)
        skf = StratifiedKFold(n_splits=5, shuffle=False)
        scorer_dic, refit = ScoreUtils.get_scorers_info()

        clf = RandomizedSearchCV(pipleine_dic[list(pipleine_dic.keys())[0]], parameters_distrib, n_iter=n_iter, cv=skf,
                                 n_jobs=1,
                                 scoring=scorer_dic, refit=refit, return_train_score=False)
        clf.fit(x_train, y_train)
        predicted_train = clf.predict(x_train)
        predicted_test = clf.predict(x_test)

        train_metrics_dic, test_metrics_dic = ScoreUtils.metrics_getter(y_train, y_test, predicted_train,
                                                                        predicted_test)
        print("Accuracy score on train: {}".format(train_metrics_dic['acc_score']))
        print("Accuracy score on test: {}".format(test_metrics_dic['acc_score']))


if __name__ == "__main__":
    # ClassifierTrainer(utils.TRAIN_INPUT_FILE, utils.TRAIN_OUTPUT_FILE).unique_clf()
    ClassifierTrainer(utils.TRAIN_INPUT_FILE, utils.TRAIN_OUTPUT_FILE).search_cv()
    # ClassifierTrainer(utils.TRAIN_INPUT_FILE, utils.TRAIN_OUTPUT_FILE).random_cv()
