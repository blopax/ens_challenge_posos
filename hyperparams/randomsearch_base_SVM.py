from sklearn import naive_bayes, svm, dummy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

from scipy.stats import uniform as sp_uniform
from scipy.stats import expon as sp_expon


n_iter = 10

dict_clf = {
    'SVM': {
        'clf': (svm.SVC(C=100, kernel='rbf', gamma=0.03, class_weight='balanced',
                        probability=True, max_iter=100000),),
        'clf__kernel': tuple(['rbf', 'linear']),
        'clf__C': sp_uniform(loc=10 ** -3, scale=10 ** 0),  # tuple([a * 10 ** i for i in range(-3, 4) for a in [1, 3]]),
        'clf__gamma': sp_uniform(loc=10 ** -3, scale=10 ** 4),  # tuple([a * 10 ** i for i in range(-3, 4) for a in [1, 3]]),
    },
}

dict_vect = {
    'union__text_transform__vect': (TfidfVectorizer(ngram_range=(1, 4),
                                                    strip_accents='unicode', analyzer='word'),),
    'union__text_transform__vect__ngram_range': tuple([(range_min, range_max) for range_min in range(1, 3)
                                                       for range_max in range(range_min, range_min + 3)]),
}
#
dict_feat_sel = {
    'union__text_transform__feat_sel': (SelectKBest(chi2, k=2000),),
    'union__text_transform__feat_sel__k': tuple([100 * a for a in range(8, 16)]),
    'union__transformer_weights': tuple([{'text_transform': 1, 'text_analysis': 0.1 * (b + 1)} for b in range(10)]),
}
