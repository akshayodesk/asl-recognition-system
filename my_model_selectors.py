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
    
    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str, n_constant=3,
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
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        #setting starting parameters
        best_num_states = 0
        score = 1e400
        N = self.X.shape[0]
        #iterate through all possible number of sates given in the beginning
        for state in range(self.min_n_components, self.max_n_components+1):
            
            try:
                #creating our HMM with current number of satets
                best_model_yet = self.base_model(state)
                #getting score for given word
                logL = best_model_yet.score(self.X, self.lengths)
               
                #calculating p - number of free parameters in HMM can be hard to find
                #These two posts helped me to get my head around finding those free params and what free params in HMM are
                #https://stats.stackexchange.com/questions/12341/number-of-parameters-in-markov-model
                #https://discussions.udacity.com/t/verifing-bic-calculation/246165/2
                p = (state * state) + ( 2 * state * N )- 1

                #Log of number of samples
                logN = np.log(N) 
                #Calculating Bayesian information criteria used to determine which model is best
                BIC = -2 * logL + p * logN
                
                #comparing current best with new BIC
                if BIC < score:
                    score = BIC
                    best_num_states = state
                
                
            except Exception:
                pass
        
        #if we couldn't calculate better number of states we set 3 as default
        if best_num_states == 0:
            best_num_states = 3

        #return HMM model with the best number of states
        return self.base_model(best_num_states)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        #Firstly lets create loop to tet models for every sigle number of components in range
        dic = -1e400 #-inf
        states_hmm_best = 0 

        #going through all possible number of states for our HMM from given range at the beginning
        for state in range(self.min_n_components, self.max_n_components+1):
            
            logPofSum = 0
            M = 0 #M is number of samples in sum 
            #Calculating SUM of all probs from the model
            try:
                #creating our model with current number of states
                model = self.base_model(state)
            except Exception:
                pass

            #because we are summing over all words except one that we are interested in we need to set checking flag
            flag = self.this_word
            for word in self.words:
                #Here we are makeing sure that we are summing over all but i (i = this_word)
                if word != flag:
                    
                    current_X, current_len = self.hwords[word]
                    
                    M += 1
                    try:
                        #increasing sum for the LogLikelihood of model given current word
                        logPofSum += model.score(current_X, current_len)
                    except:
                        pass
             
            # Until this moment we have everything except model score for word that we want to know
            try:
                #getting score of the model given this word
                logL = model.score(self.X, self.lengths)
                #getting current_score with formula for DIC
                current_score = logL - (1/(M-1))*logPofSum
            except Exception:
                #in the case that HMM couldn't calculate logL, set current_score to 0 which will prevent Selector to change the current best model
                current_score = 0
            
            
            #if current model score is worse than newly calculated score we simply change models                        
            if current_score > dic:
                dic = current_score
                states_hmm_best = state
        #when we converge (find the best number of states for our HMM) we return number of HMM with that number of states
        return self.base_model(states_hmm_best)
                
          
                


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        #starting with default parameters (worst possible)
        best_num_states = 0
        best_score = -1e400
        #n splits is variable which is helping us to determine what number of splits to find, it is going to be min of 3 and number of samples
        n_splits = min(len(self.sequences), 3)

        #going through all possible states for our HMM
        for state in range(self.min_n_components, self.max_n_components+1):

            #creating our HMM with that particular number of states
            try:
                #creating our model with current number of states
                model = self.base_model(state)
            

                #creating KFold object with n_splits as the number of splits to perform on our data
                folds = KFold(n_splits)
                #counted is used for averaging our score
                counter = 0 
                avg_score = 0

                #geting indices for train set and test set for each split 
                for cv_train_idx, cv_test_idx in folds.split(self.sequences):

                    #getting data for training and testing
                    X_train, X_train_len = combine_sequences(cv_train_idx, self.sequences)
                    X_test, X_test_len =  combine_sequences(cv_test_idx, self.sequences)

                    #training model on those folds which are ment to be for training
                    model = model.fit(X_train, X_train_len)
                    #getting log likelihood for testing data
                    logL = model.score(X_test, X_test_len)

                    avg_score += logL
                    counter += 1
                   
                #getting average score for current model
                avg_score_fin = avg_score / counter

                
                if avg_score_fin > best_score:
                    best_score = avg_score_fin
                    best_num_states = state
            
            except Exception:
                pass

        if best_num_states == 0:
            best_num_states = 3

        return self.base_model(best_num_states)