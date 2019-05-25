from sklearn import naive_bayes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2

from scipy.stats import expon as sp_expon


n_iter = 10

dict_clf = {
    # 'Dummy': {
    #     'clf': (dummy.DummyClassifier(strategy="stratified", random_state=0),)
    # },
    # 'SVM': {
    #     'clf': (svm.SVC(C=100, kernel='rbf', gamma=0.03, class_weight='balanced',
    #                     probability=True),),
    #     'clf__kernel': tuple(['rbf', 'sigmoid', 'linear']),
    #     'clf__degree': (3,),
    #     'clf__C': tuple([a * 10 ** i for i in range(-3, 4) for a in [1, 3]]),
    #     'clf__gamma': tuple([a * 10 ** i for i in range(-3, 4) for a in [1, 3]]),
    # },
    # 'nuSVC': {
    #     'clf': (svm.NuSVC(class_weight='balanced', kernel='rbf', nu=0.3, gamma=0.011,
    #                       probability=True),),
    #     'clf__kernel': tuple(['rbf', 'sigmoid', 'linear', 'poly']),
    #     'clf__degree': (5,),
    #     'clf__nu': tuple([a * 10 ** -2 for a in range(10, 31)]),
    #     'clf__gamma': tuple([a * 10 ** -3 for a in range(1, 20)]),
    # },
    # 'BernouilliNB': {
    #     'clf': (naive_bayes.BernoulliNB(alpha=0.5),),
    #     # 'clf__alpha': tuple([(a + 1) * 10 ** i for i in range(-5, 0) for a in range(1)]), # grid search
    #     'clf__alpha': sp_expon(0, 5)  # [0.0001 * (i + 1) for i in range(100)] #random
    #     # 'clf__fit_prior': (True, False)
    # },
    'MultinomialNB': {
        'clf': (naive_bayes.MultinomialNB(alpha=0.2),),
        # 'clf__alpha': tuple([a * 10 ** i for i in range(-5, 0) for a in range(10)]),
        'clf__alpha': sp_expon(0, 5),  # [0.0001 * (i + 1) for i in range(100)] #random
        'clf__fit_prior': (True, False)
    }
}

dict_vect = {
    'vect': (TfidfVectorizer(use_idf=True, smooth_idf=True, ngram_range=(1, 4)),),
    # 'vect__use_idf': (True,),
    # 'vect__smooth_idf': (True,),
    'vect__ngram_range': tuple([(range_min, range_max) for range_min in range(1, 2)
                               for range_max in range(range_min, range_min + 2)]),
}

dict_feat_sel = {
    'feat_sel': (SelectKBest(chi2, k=2000),),
    'feat_sel__k': (1000, 2000, 3000)
}
