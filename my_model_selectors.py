import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def scoreBIC(self, logLike, number_of_parameters, number_of_data_points):
        return ((-2 * logLike) + (number_of_parameters * np.log(number_of_data_points)))

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores

        model_list = {}
        n_components = self.min_n_components

        while n_components >= self.min_n_components and n_components <=self.max_n_components:
            hmm_model = self.base_model(n_components)
            try:
                test_score = hmm_model.score(self.X, self.lengths)
                bic = self.scoreBIC(test_score, n_components, len(self.X))
                model_list[hmm_model] = bic
            except:
                pass
            n_components = n_components + 1

        # https://discussions.udacity.com/t/hmmlearn-valueerror-rows-of-transmat--must-sum-to-1-0/229995/5?u=iamsteveng
        # Handle this problem
        if len(model_list) == 0:
            return self.base_model(self.n_constant)

        result_model = max(model_list, key=lambda i: model_list[i])
        print("[SelectorBIC] Result model n_components: {}".format(result_model.n_components))
        return result_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def calculate_avgscore_otherwords (self, except_this_word, hmm_model):
        average_score = 0
        num_words = len(self.words)
        num_words = num_words - 1
        for word in self.words:
            if word == except_this_word:
                continue
            print(".", end="", flush=True)
            this_X, this_lengths = self.hwords[word]
            try:
                score = hmm_model.score(this_X, this_lengths)
                average_score = average_score + score
            except:
                num_words = num_words -1

        # Calculate the average score
        if num_words > 0:
            average_score = average_score/num_words
        else:
            average_score = 0
        return average_score

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores

        model_list = {}
        n_components = self.min_n_components

        print("DIC selector training: ", end="")
        while n_components >= self.min_n_components and n_components <= self.max_n_components:
            print("{}, ".format(n_components), end="", flush=True)
            hmm_model = self.base_model(n_components)
            try:
                this_word_score = hmm_model.score(self.X, self.lengths)
                otherwords_avgscore = self.calculate_avgscore_otherwords(self.this_word, n_components)
                diff = this_word_score - otherwords_avgscore
                model_list[hmm_model] = diff
            except:
                pass
            n_components = n_components + 1

        print("Finished", flush=True)

        if len(model_list) == 0:
            return self.base_model(self.n_constant)

        result_model = max(model_list, key=lambda i: model_list[i])
        print("[SelectorDIC] Result model n_components: {}".format(result_model.n_components))
        return result_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV

        N_SPLITS = 3
        split_method = KFold(n_splits=N_SPLITS)
        model_list = {}
        n_components = self.min_n_components

        try:
            while n_components >= self.min_n_components and n_components <=self.max_n_components:
                average_test_score = 0
                hmm_model = None
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    # print("[SelectorCV] Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))
                    self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                    hmm_model = self.base_model(n_components)
                    test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)
                    test_score = hmm_model.score(test_X, test_lengths)
                    # print("[SelectorCV] logL:{}".format(test_score))
                    average_test_score = average_test_score + test_score
                average_test_score = average_test_score/N_SPLITS
                model_list[hmm_model] = average_test_score
                n_components = n_components + 1
        except:
            # print("[SelectorCV] Cannot use KFold")
            hmm_model = self.base_model(self.n_constant)
            # test_score = hmm_model.score(self.X, self.lengths)
            # print("[SelectorCV] logL:{}".format(test_score))
            return hmm_model

        result_model = max(model_list, key=lambda i: model_list[i])
        print("[SelectorCV] Result model n_components: {}".format(result_model.n_components))
        return result_model
