"""
Original Paper Citation:
M. Gaur, A. Alambo, J. P. Sain, U. Kursuncu, K. Thirunarayan, R. Kavuluru, A. Sheth, R. Welton, and J. Pathak, 
“Knowledge-aware assessment of severity of suicide risk for early intervention,” The World Wide Web Conference, 2019. 

Original Paper Code Repo: https://github.com/jpsain/Suicide-Severity

Referenced pre-processing code from the following implementation of the paper: https://github.com/rreube3/CSE6250/blob/main/Create_External_Features.py
"""
import nltk
import pandas as pd
import numpy as np
import statistics
import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

first_person = ["i", "me", "my", "mine", "we", "us", "our", "ours"]
other_person = ["you", "your", "yours", "he", "she", "him", "her", "his", "hers", "they", "them", "their", "theirs", "youll", "youd"]
pattern = [{'POS': 'VERB', 'OP': '?'},
            {'POS': 'ADV', 'OP': '*'},
            {'POS': 'AUX', 'OP': '*'},
            {'POS': 'VERB', 'OP': '+'}]

# You must download the spacy library and "en_core_web_sm" database.
# You can use the command to download it: python3 -m spacy download en_core_web_sm
en_core_web_sm = spacy.load("en_core_web_sm", disable = ['ner'])
matcher = Matcher(en_core_web_sm.vocab)
matcher.add("Verb phrase", [pattern])
        
class ExternalFeatures():
    def __init__(self):
        self.raw_data = pd.read_csv("data/500_Reddit_users_posts_labels.csv")
        afinn_data = pd.read_csv('data/AFINN-en-165.txt', sep="\t", header=None)
        labMT_data = pd.read_csv("data/labMT")

        self.afinn_dict = dict(zip(afinn_data[0], afinn_data[1]))
        self.twit_dict = dict(zip(labMT_data["word"], labMT_data["twitter_rank"]))
        self.goog_dict = dict(zip(labMT_data["word"], labMT_data["google_rank"]))
        self.nyt_dict = dict(zip(labMT_data["word"], labMT_data["nyt_rank"]))
        self.lyr_dict = dict(zip(labMT_data["word"], labMT_data["lyrics_rank"]))

        self.hrank_dict = dict(zip(labMT_data["word"], labMT_data["happiness_rank"]))
        self.havg_dict = dict(zip(labMT_data["word"], labMT_data["happiness_average"]))
        self.hstdv_dict = dict(zip(labMT_data["word"], labMT_data["happiness_standard_deviation"]))

    def preprocess_text(self, raw_text):
        forbidden_chars = ",.:;'\"-?!/"
        allowed_chars = 'abcdefghijklmnopqrstuvwxyz '
        processed_text = raw_text.lower()
        for char in forbidden_chars:
            processed_text = processed_text.replace(char, ' ')

        filtered_text = ''.join(filter(lambda x: x in allowed_chars, processed_text))

        return filtered_text

    def calculate_score(self, text, oper):
        cleaned_text = self.preprocess_text(text)
        temp = []

        for x in cleaned_text.split():
            if oper == "rank" and x in self.hrank_dict.keys():
                temp.append(self.hrank_dict[x])

            elif oper == "avg" and x in self.havg_dict.keys():
                temp.append(self.havg_dict[x])

            elif oper == "std" and x in self.hstdv_dict.keys():
                temp.append(self.hstdv_dict[x])

            elif oper == "twitter" and x in self.twit_dict.keys() and self.twit_dict[x] == self.twit_dict[x]: 
                temp.append(self.twit_dict[x])
            
            elif oper == "google" and x in self.goog_dict.keys() and self.goog_dict[x] == self.goog_dict[x]:
                temp.append(self.goog_dict[x])

            elif oper == "nyt" and x in self.nyt_dict.keys() and self.nyt_dict[x] == self.nyt_dict[x]:
                temp.append(self.nyt_dict[x])

            elif oper == "lyric" and x in self.lyr_dict.keys() and self.lyr_dict[x] == self.lyr_dict[x]:
                temp.append(self.lyr_dict[x])

        return statistics.mean(temp)

    def calculate_avg_tree_depth(self, text):
        # Referenced: https://gist.github.com/drussellmrichie/47deb429350e2e99ffb3272ab6ab216a
        def height(root):
            if list(root.children):
                return 1 + max(height(x) for x in root.children)

            else:
                return 1

        def avg_heights(para):
            if type(para) == str:
                temp = en_core_web_sm(para)

            else:
                temp = para
                
            roots = []
            for x in temp.sents:
                roots.append(x.root)

            return np.mean([height(r) for r in roots])

        return avg_heights(text)

    def calculate_max_verb_phrase(self, text):
        # Referenced: https://stackoverflow.com/questions/47856247/extract-verb-phrases-using-spacy
        doc = en_core_web_sm(text)
        match = matcher(doc)
        temp = [doc[start:end] for _, start, end in match]
        max = 0

        for x in filter_spans(temp):
            if str(x).count(" ") > max:
                max = str(x).count(" ")

        return max + 1

    def count_fpp(self, text):
        cleaned_text = self.preprocess_text(text)
        
        count = 0
        for x in cleaned_text.split():
            if x in first_person:
                count += 1
        
        return count

    def sentence_count(self, text):
        return len(sent_tokenize(text))

    def articles_count(self, text):
        doc = en_core_web_sm(text)
        count = 0

        for np in doc.noun_chunks:
            lower_text = str(np).lower()
            if "the " in lower_text:
                count += 1

        return count

    def calculate_fpp_ratio(self, text):
        cleaned_text = self.preprocess_text(text)
        first = 0
        other = 0.1
        for x in cleaned_text.split():
            if x in other_person:
                other += 1

            elif x in first_person:
                first += 1
        
        return first / other
    
    def calculate_afinn_score(self, text):
        cleaned_text = self.preprocess_text(text)
        scores = []
        for x in cleaned_text.split():
            if x in self.afinn_dict.keys():
                scores.append(self.afinn_dict[x])
        
        if len(scores) <= 0:
            return 0

        return statistics.mean(scores)

    def run(self):
        self.raw_data["AFINN_score"] = self.raw_data["Post"].apply(self.calculate_afinn_score)
        self.raw_data["FPP_ratio"] = self.raw_data["Post"].apply(self.calculate_fpp_ratio)
        self.raw_data["twit_score"] = self.raw_data["Post"].apply(self.calculate_score, oper="twitter")
        self.raw_data["goog_score"] = self.raw_data["Post"].apply(self.calculate_score, oper="google")
        self.raw_data["nyt_score"] = self.raw_data["Post"].apply(self.calculate_score, oper="nyt")
        self.raw_data["lyric_score"] = self.raw_data["Post"].apply(self.calculate_score, oper="lyric")

        self.raw_data["hrank_score"] = self.raw_data["Post"].apply(self.calculate_score, oper="rank")
        self.raw_data["havg_score"] = self.raw_data["Post"].apply(self.calculate_score, oper="avg")
        self.raw_data["hstdv_score"] = self.raw_data["Post"].apply(self.calculate_score, oper="std")

        self.raw_data["parse_tree_height"] = self.raw_data["Post"].apply(self.calculate_avg_tree_depth)
        self.raw_data["verb_phrase_length"] = self.raw_data["Post"].apply(self.calculate_max_verb_phrase)
        self.raw_data["FPP_count"] = self.raw_data["Post"].apply(self.count_fpp)
        self.raw_data["count_sentence"] = self.raw_data["Post"].apply(self.sentence_count)
        self.raw_data["count_def_articles"] = self.raw_data["Post"].apply(self.articles_count)
        
        final_data = self.raw_data[["User", "AFINN_score", "FPP_ratio", "hrank_score", "havg_score", "hstdv_score", "twit_score", "goog_score", "nyt_score", "lyric_score", "parse_tree_height", "verb_phrase_length", "FPP_count", "count_sentence", "count_def_articles"]]
        final_data.to_csv("data/External_Features.csv", index = False)

external_features = ExternalFeatures()
external_features.run()