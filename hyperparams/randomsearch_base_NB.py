from sklearn import naive_bayes, svm, dummy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

from scipy.stats import uniform as sp_uniform
from scipy.stats import expon as sp_expon


# For NB: Accuracy score on train: 0.7435417446649195
# Accuracy score on test: 0.5305676855895196


n_iter = 10

dict_clf = {
    'BernouilliNB': {
        'clf': (naive_bayes.BernoulliNB(alpha=0.5),),
        'clf__alpha': sp_expon(0, 5),  # [0.0001 * (i + 1) for i in range(100)] #random
        'clf__fit_prior': (True, False)
    },
    'MultinomialNB': {
        'clf': (naive_bayes.MultinomialNB(alpha=0.2),),
        'clf__alpha': sp_expon(0, 5),  # [0.0001 * (i + 1) for i in range(100)] #random
        'clf__fit_prior': (True, False)
    }
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
