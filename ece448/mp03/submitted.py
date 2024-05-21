'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.

For implementation of this MP, You may use numpy (though it's not needed). You may not 
use other non-standard modules (including nltk). Some modules that might be helpful are 
already imported for you.
'''

import math
from collections import defaultdict, Counter
from math import log
import numpy as np

# define your epsilon for laplace smoothing here

def baseline(test, train):
    '''
    Implementation for the baseline tagger.
    input:  test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
            training data (list of sentences, with tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
#       raise NotImplementedError("You need to write this part!")
    # get all the tag and word to a dic
    # count the tag for the same time for the unknown
    word_list = {}
    tag_count = Counter()
    for sentence in train:
        for word, tag in sentence:
            if word not in word_list:
                word_list[word] = Counter()
            tag_count[tag] += 1
            word_list[word][tag] += 1
    # give every word the most common tag
    word_tag = {}
    for word in word_list.keys():
        most_common_element = word_list[word].most_common(1)
        tag = most_common_element[0][0]
        word_tag[word] = tag
    most_common_element = tag_count.most_common(1)
    notseen_tag = most_common_element[0][0]
    # get the output
    output =  []
    for sentence in test:
        new_sentence = []
        for word in sentence:
            if word in word_tag:
                new_sentence.append((word,word_tag[word]))
            else:
                new_sentence.append((word,notseen_tag))
        output.append(new_sentence)
    return output
            
def viterbi(test, train):
    '''
    Implementation for the viterbi tagger.
    input:  test data (list of sentences, no tags on the words)
            training data (list of sentences, with tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # for smoothing
    k = 1e-5
    # get the tag and word relationship first
    tag_count = Counter()
    tag_pair = defaultdict(Counter)
    tag_word = defaultdict(Counter)

    for sentence in train:
        pre_tag = 'START'
        for word,tag in sentence:
            tag_count[tag] += 1
            tag_pair[pre_tag][tag] += 1
            tag_word[tag][word] += 1
            pre_tag = tag

    # get the init pr
    init_pr = defaultdict()
    for tag in tag_pair['START'].keys():
        pr = (tag_pair['START'][tag] + k) / (k * len(tag_count) + tag_count['START'])
        init_pr[tag] = log(pr)
    
    # get the transition pr
    tran_pr = defaultdict(lambda: defaultdict())
    for tag1 in tag_count:
        for tag2 in tag_count:
            pr = (tag_pair[tag1][tag2] + k) / (tag_count[tag1] + len(tag_count) * k)
            tran_pr[tag1][tag2] = log(pr)
    
    # get the emission pr
    emi_pr = defaultdict(lambda: defaultdict())
    for tag in tag_count:
        for word in tag_word[tag]:
            # add 1 for the unknown, using ' ' to represent
            pr = (tag_word[tag][word] + k) / (tag_count[tag] + k * (len(tag_word[tag]) + 1)) 
            emi_pr[word][tag] = log(pr)
        emi_pr[' '][tag] = log(k / (tag_count[tag] + k * (len(tag_word[tag]) + 1)))

    # construct the trellis and get the best path
    output = []
    for sentence in test:
        new_sentence = trellis_result(init_pr,tran_pr,emi_pr,sentence)
        output.append(new_sentence)
    return output

def trellis_result(init_pr,tran_pr,emi_pr,sentence):
    new_sentence = []
    pr = defaultdict(lambda: defaultdict())
    # tag_back[tag][n] save the corresponding tag for the n-1 node
    tag_back = defaultdict( lambda: defaultdict())
    tag_list = []
    # first the init
    for tag in init_pr.keys():
        if sentence[0] not in emi_pr[tag].keys():
            pr[0][tag] = init_pr[tag] + emi_pr[' '][tag]
        else:
            pr[0][tag] = init_pr[tag] + emi_pr[sentence[0]][tag]
    # iteration
    for index in range(len(sentence)):
        if index == 0:
            continue
        word = sentence[index]
        if word not in emi_pr.keys():
            word = ' '
        for tag in emi_pr[word].keys():
            tem_pr = -float('inf')
            tem_tag = None
            for pre_tag in pr[index - 1].keys():
                cur_pr = pr[index - 1][pre_tag] + tran_pr[pre_tag][tag] + emi_pr[word][tag]
                if cur_pr > tem_pr:
                    tem_pr = cur_pr
                    tem_tag = pre_tag
            pr[index][tag] = tem_pr
            tag_back[index][tag] = tem_tag
                     
    # termination
    final_pr = -float('inf')
    final_tag = None
    for tag in pr[len(sentence) - 1]:
         cur_pr = pr[len(sentence) - 1][tag]
         if cur_pr > final_pr:
            final_pr = cur_pr
            final_tag = tag

    # back-trace
    tag_temp = final_tag
    tag_list.append(final_tag)
    for index in range(len(sentence)):
        if len(sentence) - index - 1 == 0:
            break
        tag_list.append(tag_back[len(sentence) - index - 1][tag_temp])
        tag_temp = tag_back[len(sentence) - index - 1][tag_temp]
    tag_list.reverse()
    for index in range(len(sentence)):
        new_sentence.append((sentence[index],tag_list[index]))
    return new_sentence

def viterbi_ec(test, train):
    '''
    Implementation for the improved viterbi tagger.
    input:  test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
            training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    k = 1e-5
    # get the tag and word relationship first
    tag_count = Counter()
    tag_pair = defaultdict(Counter)
    tag_word = defaultdict(Counter)
    word_tag = defaultdict()
    word_count = Counter()
    for sentence in train:
        pre_tag = 'START'
        for word,tag in sentence:
            tag_count[tag] += 1
            tag_pair[pre_tag][tag] += 1
            tag_word[tag][word] += 1
            word_tag[word] = tag
            word_count[word] += 1
            pre_tag = tag

    # get the init pr
    init_pr = defaultdict()
    for tag in tag_pair['START'].keys():
        pr = (tag_pair['START'][tag] + k) / (k * len(tag_count) + tag_count['START'])
        init_pr[tag] = log(pr)
    
    # get the transition pr
    tran_pr = defaultdict(lambda: defaultdict())
    for tag1 in tag_count:
        for tag2 in tag_count:
            pr = (tag_pair[tag1][tag2] + k) / (tag_count[tag1] + len(tag_count) * k)
            tran_pr[tag1][tag2] = log(pr)
    
    # get the emission pr (optimize here!)
    # first get the hapax word
    hapax_word_tag = {}
    hapax_tag_list = {}
    for word in word_count:
        if word_count[word] == 1:
            tag = word_tag[word]
            hapax_word_tag[word] = tag
    # print(hapax_word_tag.values())
    for tag in hapax_word_tag.values():
        if tag not in hapax_tag_list:
            hapax_tag_list[tag] = 0
        hapax_tag_list[tag] += 1
    # print(hapax_tag_list)
    # print(len(tag_count))
    # calculate the optimized emission pr
    emi_pr = defaultdict(lambda: defaultdict())
    for tag in tag_count:
        hapax = 0
        if tag in hapax_tag_list:
            hapax = hapax_tag_list[tag] / (sum(hapax_tag_list.values()) + k * (len(hapax_tag_list) + 1))
        else:
            hapax = k / (sum(hapax_tag_list.values()) + k * (len(hapax_tag_list) + 1))
        for word in tag_word[tag]:
            # add 1 for the unknown, using ' ' to represent
            pr = (tag_word[tag][word] + k * hapax) / (tag_count[tag] + k * hapax * (len(tag_word[tag]) + 1)) 
            emi_pr[word][tag] = log(pr)
        emi_pr[' '][tag] = log(k * hapax / (tag_count[tag] + k * hapax * (len(tag_word[tag]) + 1)))
    # construct the trellis and get the best path
    output = []
    for sentence in test:
        new_sentence = trellis_result(init_pr,tran_pr,emi_pr,sentence)
        output.append(new_sentence)
    return output



