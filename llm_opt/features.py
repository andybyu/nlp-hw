# Jordan Boyd-Graber
# 2023
#
# Feature extractors to improve classification to determine if an answer is
# correct.

from collections import Counter
from math import log
from numpy import mean
import gzip
import json

class Feature:
    """
    Base feature class.  Needs to be instantiated in params.py and then called
    by buzzer.py
    """

    def __init__(self, name):
        self.name = name

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        """

        question -- The JSON object of the original question, you can extract metadata from this such as the category

        run -- The subset of the question that the guesser made a guess on

        guess -- The guess created by the guesser

        guess_history -- Previous guesses (needs to be enabled via command line argument)

        other_guesses -- All guesses for this run
        """


        raise NotImplementedError(
            "Subclasses of Feature must implement this function")

    
"""
Given features (Length, Frequency)
"""
class LengthFeature(Feature):
    """
    Feature that computes how long the inputs and outputs of the QA system are.
    """

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        # How many characters long is the question?

        # guess_length = len(guess)
        run_length = len(run)
        guess_length = len(guess)

        # How many words long is the question?
        # How many characters long is the run?
        if run is None or run=="":  
            yield ("run", -1)         
        else:                           
            yield ("run", run_length)
        
        if guess is None or guess=='':
            yield ('guess', -1)
        else:
            yield ('guess', guess_length)
            
class StandardizedLengthFeature(Feature):
    def __init__(self, name):
        from eval import normalize_answer
        self.name = name
        self.normalize = normalize_answer
        self.char_lengths = []
        self.word_lengths = []
        
    def add_training(self, question_source):
        import json
        with gzip.open(question_source) as infile:
            questions = json.load(infile)
        for ii in questions:
            ii_split = ii['text'].split(' ')
            self.char_lengths.append(len(ii['text']))
            self.word_lengths.append(len(ii_split))
            
    def __call__(self, question, run, guess, guess_history, guesses):
        # We only use question, run, and guess (same as before)
        # guess_history and guesses are ignored since we don't need them
        import numpy as np
        yield('std_char_length', (len(run) - np.mean(self.char_lengths)) / np.var(self.char_lengths))
        yield('std_word_length', (len(run.split(' ')) - np.mean(self.word_lengths)) / np.var(self.word_lengths))
        yield('guess_length', len(guess))
        
    
        
        
class GuessBlankFeature(Feature):
    """
    Is guess blank?
    """
    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        yield ('true', len(guess) == 0)


class GuessCapitalsFeature(Feature):
    """
    Capital letters in guess
    """
    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        yield ('true', log(sum(i.isupper() for i in guess) + 1))
        

# Somehow latch on to the indicator? Will rule out things like this
class NumberThisFeature(Feature):
    """
    Returns the number of occurrences of "this" or "these" in the run
    """
    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        run_split = run.split(' ')
        count = 0
        for word in run_split:
            # Latch on to the first word after "this" or "these"
            if word == 'this':
                count += 1
        yield ('NumberThis', count)
                
class SentencesReadFeature(Feature):
    """
    Returns how many sentences have been read divided by number of total sentences in the question
    (crudely defined as the number of periods in the run/question respectively.)
    """
    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        
        r_count = 0
        
        for char in run:
            if char == '.':
                r_count += 1
        
        yield ('SentencesRead', r_count)

class AppearedInRunFeature(Feature):
    """
    Returns whether the guess has appeared in the run or not
    """
    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        yield ('in_run', int(guess.lower() in run.lower()))
    # Ex. prevents a guess of "saga" when saga is already heard

class IndicatorFeature(Feature):
    """
    Latch on to the first word that appears in the run after the word "this".
    Look through the training data and see if there are any questions with the indicator `indicator` and the answer `guess`.
    Return 1 if true, return 0 if false
    """
    def __init__(self, name):
        from eval import normalize_answer
        self.name = name
        self.ia_pairs = []
        self.normalize = normalize_answer
    
    def get_indicator(self, text):
        import re
        """
        Latches on to the first word that comes after this or these, and returns that as the "indicator"
        """
        text_split = text.lower().split(' ')
        for i in range(len(text_split)-1):
            word = text_split[i]
            if word == 'this' or word == 'these':
                indicator = text_split[i+1].rstrip("'s") # Crude removal of possessive case
                indicator = re.sub(r'[^a-zA-Z]+$', '', indicator)
                return indicator
        return ''
    
    def add_training(self, question_source):
        import json
        with gzip.open(question_source) as infile:
            questions = json.load(infile)
        for ii in questions:
            self.ia_pairs.append((self.get_indicator(ii['text']), self.normalize(ii['answer'])))
    
    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        to_return = ('indicator', 0)
        indicator = self.get_indicator(run)
        normalized_guess = self.normalize(guess)
        for i, a in self.ia_pairs:
            if (normalized_guess in a) and indicator == i:
                to_return = ('indicator', 1)
                break
        
        yield to_return
            
        
        # Now look through the training data. 
        # Do any questions whose answer is `guess` have a question with the indicator of `indicator`?
        # If so return 1, if not return 0
        
class FrequencyFeature:
    def __init__(self, name):
        from eval import normalize_answer
        self.name = name
        self.counts = Counter()
        self.normalize = normalize_answer
        
    def add_training(self, question_source):
        import json
        with gzip.open(question_source) as infile:
            questions = json.load(infile)
        for ii in questions:
            self.counts[self.normalize(ii["page"])] += 1
            
    def __call__(self, question, run, guess, guess_history, guesses):
        # We only use question, run, and guess (same as before)
        # guess_history and guesses are ignored since we don't need them
        
        frequency_value = log(1 + self.counts[self.normalize(guess)])
        yield ("guess", frequency_value)
    

if __name__ == "__main__":
    """

    Script to write out features for inspection or for data for the 470
    logistic regression homework.

    """
    import argparse
    
    from parameters import add_general_params, add_question_params, \
        add_buzzer_params, add_guesser_params, setup_logging, \
        load_guesser, load_questions, load_buzzer

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_guess_output', type=str)
    add_general_params(parser)    
    guesser_params = add_guesser_params(parser)
    buzzer_params = add_buzzer_params(parser)    
    add_question_params(parser)

    flags = parser.parse_args()

    setup_logging(flags)

    guesser = load_guesser(flags, guesser_params)
    buzzer = load_buzzer(flags, buzzer_params)
    questions = load_questions(flags)

    buzzer.add_data(questions)
    buzzer.build_features(flags.buzzer_history_length,
                          flags.buzzer_history_depth)

    vocab = buzzer.write_json(flags.json_guess_output)
    with open("data/small_guess.vocab", 'w') as outfile:
        for ii in vocab:
            try:
                outfile.write("%s\n" % ii)
            except UnicodeEncodeError:
                outfile.write("%s\n" % unidecode(ii))
