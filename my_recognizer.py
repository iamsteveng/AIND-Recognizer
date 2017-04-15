import warnings
from asl_data import SinglesData
from hmmlearn.hmm import GaussianHMM


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

    print("Recognizing words: ", end="")
    for word_id in range(0,len(test_set.get_all_Xlengths())):
        current_X, current_lengths = test_set.get_item_Xlengths(word_id)
        test_word_probability_dict = {}
        # Calculate log likelihood against each word model
        for model_word, model in models.items():
            try:
                score = model.score(current_X, current_lengths)
                test_word_probability_dict[model_word] = score
            except:
                test_word_probability_dict[model_word] = float("-inf")
        # Append the probability list
        probabilities.append(test_word_probability_dict)
        # Get the word of largest log likelihood
        guesses.append(max(test_word_probability_dict, key=lambda i: test_word_probability_dict[i]))
        print(".", end="")
    print(" Finished")

    # return probabilities, guesses
    return (probabilities, guesses)
