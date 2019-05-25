from sklearn.metrics import make_scorer, log_loss
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, matthews_corrcoef, f1_score,\
    precision_score, recall_score


class ScoreUtils:
    def __init__(self):
        pass

    @staticmethod
    def matthews_scorer():
        return make_scorer(matthews_corrcoef)

    # @staticmethod
    # def balanced_accuracy_scorer():
    #    return make_scorer(balanced_accuracy_score)

    @staticmethod
    def log_loss_scorer():
        return make_scorer(log_loss, greater_is_better=False, needs_proba=True)

    @staticmethod
    def get_scorers_info():
        scorer_dic = {
            'accuracy': 'accuracy',
            # 'balanced_acc': ScoreUtils.balanced_accuracy_scorer(),
            'matthews': ScoreUtils.matthews_scorer(),
            # 'log_loss': ScoreUtils.log_loss_scorer()
        }
        refit = 'matthews'
        return scorer_dic, refit

    @staticmethod
    def metrics_getter(y_train, y_test, predicted_train, predicted_test):
        train_metrics_dic = {
            'class_report': classification_report(y_train, predicted_train),
            'matrix': confusion_matrix(y_train, predicted_train),
            'acc_score': accuracy_score(y_train, predicted_train),
            'matt_corrcoef': matthews_corrcoef(y_train, predicted_train),
            'f1_score': f1_score(y_train, predicted_train, average='weighted'),
            'precision_score': precision_score(y_train, predicted_train, average='weighted'),
            'recall_score': recall_score(y_train, predicted_train, average='weighted')
        }
        test_metrics_dic = {
            'class_report': classification_report(y_test, predicted_test),
            'matrix': confusion_matrix(y_test, predicted_test),
            'acc_score': accuracy_score(y_test, predicted_test),
            'matt_corrcoef': matthews_corrcoef(y_test, predicted_test),
            'f1_score': f1_score(y_test, predicted_test, average='weighted'),
            'precision_score': precision_score(y_test, predicted_test, average='weighted'),
            'recall_score': recall_score(y_test, predicted_test, average='weighted')
        }
        return train_metrics_dic, test_metrics_dic

    # todo abstract refit for search (to decide if logloss or matthews or sth else)
    # todo if confirmation of false negative create customized scorer that take it into account
    # -->(matthews formula but squaring false negative? ou MCC * recall ( * 2...)
