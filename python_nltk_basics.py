# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import nltk

from nltk.tokenize import sent_tokenize, word_tokenize

example_text = "Hello Mr. Smith, how are you doning today? The weather is great, and python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."

print(sent_tokenize(example_text))

print(word_tokenize(example_text))


# Stopwords in NLTK
from nltk.corpus import stopwords
set(stopwords.words('english'))

# Example of stopwords removal
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sent = "This is a sample sentence, showing off the stop words filteration"

stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(example_sent)

filtered_sentence = [w for w in word_tokens if not w in stop_words]

print(word_tokens)
print(filtered_sentence)


## Stemming example
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

example_words = "python pythoner pythoning pythoned pythonly".split()

for w in example_words:
    print(ps.stem(w))
    

new_text = "It is important to be very pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."
words = word_tokenize(new_text)

for w in words:
    print(ps.stem(w))
    

## Part of Speech tagging
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_set_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_set_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized[:5]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)
            
    except Exception as e:
        print(str(e))
        

process_content()





## Chinking based on POS

from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2006-GWBush.txt')

custom_set_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_set_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized[:5]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            
            chunkGram = r"""Chunk: {<.*>+}
            }<VB.?|IN|DT|TO>+{"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            chunked.draw()
            
    except Exception as e:
        print(str(e))
        
process_content()










