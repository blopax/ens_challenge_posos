from hyperparams import randomsearch_base
from hyperparams import randomsearch_base_SVM
from hyperparams import randomsearch_base_SVC
from hyperparams import randomsearch_base_NB
from hyperparams import gridsearch_base


# noinspection PyUnresolvedReferences
class ClfParamsSetter:
    """
    setting the tested classifiers and the params to be tested in hyperoptimization
    Idea is to make a file parser to get a dict with all infos that can be used
    """

    def __init__(self, mode='search'):
        if mode == 'search':
            self.n_iter = gridsearch_base.n_iter
            self.dict_clf = gridsearch_base.dict_clf
            self.dict_vect = gridsearch_base.dict_vect
            self.dict_feat_sel = gridsearch_base.dict_feat_sel
        else:  # mode =='random':
            self.n_iter = randomsearch_base.n_iter
            self.dict_clf = randomsearch_base.dict_clf
            self.dict_vect = randomsearch_base.dict_vect
            self.dict_feat_sel = randomsearch_base.dict_feat_sel

    def get_parameters(self):
        dict_parameters = dict()
        n_iter, dict_clf, dict_vect, dict_feat_sel = self.n_iter, self.dict_clf, self.dict_vect, self.dict_feat_sel

        if type(dict_clf[list(dict_clf.keys())[0]]) == dict:
            for k, v in dict_clf.items():
                dict_parameters[k] = {**(dict_clf[k]), **dict_vect, **dict_feat_sel}
        else:
            dict_parameters['clf'] = {**dict_clf, **dict_vect, **dict_feat_sel}

        parameters_list = []
        for k, v in dict_parameters.items():
            parameters_list.append(v)

        distrib = dict()
        for k, v in dict_parameters.items():
            if k in ["BernouilliNB", "MultinomialNB", 'SVM', 'nuSVC', 'Dummy']:
                for key, value in v.items():
                    distrib[key] = value
            else:
                distrib[k] = v
        return dict_parameters, parameters_list, distrib, n_iter


if __name__ == '__main__':
    dict_params, params_list, dic_clf, max_iter = ClfParamsSetter().get_parameters()
    print("dict_parameters is:\n{}\n\n parameters_list is:\n{}\n".format(dict_params, params_list))
