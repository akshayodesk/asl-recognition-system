import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    
    #get all words for comparing
    words = models.keys()

    #get all words that we are going to test our model on, 
    #we would like to have the biggest prob on word which is also our word from words set
    all_words = test_set.get_all_Xlengths()

    #run through all words 
    for _, data_for_test in all_words.items():

      #getting data for testing word
      X, lengs = data_for_test
      

      our_dics = {}
      #running through all words that we use for testing accuracy of our HMMs
      for word in words:

        model = models[word]
        try:
          #getting log likelihood of the test word 
          our_dics[word] = model.score(X, lengs)
        except:
          pass

      #this is list of all words tested for each word in our words list
      probabilities.append(our_dics)

    #getting word which has the biggest probability in each dict.
    #we want the key word to be the same as our word which was predicted by HMM
    guesses = [max(prob, key=prob.get) for prob in probabilities]

    return probabilities, guesses
