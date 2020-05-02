#-*- coding:utf-8 -*-


### Requirements

# newspaper3k
# konlpy (이미 설치됨 ㅇㅇ)
# scikit-learn

####

from newspaper import Article
from konlpy.tag import Kkma, Twitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import numpy as np


class SentenceTokenizer(object):
    def __init__(self):
        self.kkma = Kkma()
        self.twitter = Twitter()
        self.stopwords = ['중인', '만큼', '마찬가지', '꼬집었', "연합뉴스", "데일리", "동아일보", "중앙일보", "조선일보", "기자" , "아", "휴", "아이구", "아이쿠", "아이고", "어", "나", "우리", "저희", "따라", "의해", "을", "를", "에", "의", "가", ]

    def url2sentences(self, url):
        article = Article(url=url, language='ko')
        article.download()
        article.parse()
        sentences = self.kkma.sentences(article.text)

        for i in range(0, len(sentences)):
            if len(sentences[i]) <= 10:
                sentences[i-1] += (' ' + sentences[i])
                sentences[i] = ' '

        return sentences

    def text2sentences(self, text):
        sentences = self.kkma.sentences(text)

        for i in range(0, len(sentences)):
            if len(sentences[i]) <= 10:
                sentences[i - 1] += (' ' + sentences[i])
                sentences[i] = ' '

        return sentences

    def get_nouns(self, sentences):
        nouns = []

        for sentence in sentences:
            if sentence is not ' ':
                nouns.append(' '.join([noun for noun in self.twitter.nouns(str(sentence)) if noun not in self.stopwords and len(noun) > 1]))

        return nouns


class GraphMatrix(object):
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.cnt_vec = CountVectorizer()
        self.graph_sentence = []

    def build_sent_graph(self, sentence):
        tfidf_mat = self.tfidf.fit_transform(sentence).toarray()
        self.graph_sentence = np.dot(tfidf_mat, tfidf_mat.T)

        return self.graph_sentence

    def build_words_graph(self, sentence):
        cnt_vec_mat = normalize(self.cnt_vec.fit_transform(sentence).toarray().astype(float), axis=0)
        vocab = self.cnt_vec.vocabulary_

        return np.dot(cnt_vec_mat.T, cnt_vec_mat), {vocab[word] : word for word in vocab}


class Rank(object):
    def get_ranks(self, graph, d=0.85):
        A = graph
        matrix_size = A.shape[0]
        for i in range(matrix_size):
            A[i, i] = 0
            link_sum = np.sum(A[:, i])
            if link_sum != 0:
                A[:, i] /= link_sum

            A[:, i] *= -d
            A[i, i] = 1

        B = (1-d) * np.ones((matrix_size, 1))
        ranks = np.linalg.solve(A, B)

        return {idx: r[0] for idx, r in enumerate(ranks)}


class TextRank(object):
    def __init__(self, text):
        self.sent_tokenize = SentenceTokenizer()

        if text[:5] in ('http:', 'https'):
            self.sentences = self.sent_tokenize.url2sentences(text)
        else:
            self.sentences = self.sent_tokenize.text2sentences(text)

        print("=== Tokenized Setences ===")
        for x in self.sentences:
            print(x)
        print("=========================\n")

        self.nouns = self.sent_tokenize.get_nouns(self.sentences)
        self.graph_matrix = GraphMatrix()

        self.sent_graph = self.graph_matrix.build_sent_graph(self.nouns)
        self.words_graph, self.idx2word = self.graph_matrix.build_words_graph(self.nouns)

        self.rank = Rank()
        self.sent_rank_idx = self.rank.get_ranks(self.sent_graph)
        self.sorted_sent_rank_idx = sorted(self.sent_rank_idx, key=lambda k: self.sent_rank_idx[k], reverse=True)

        self.word_rank_idx = self.rank.get_ranks(self.words_graph)
        self.sorted_word_rank_idx = sorted(self.word_rank_idx, key=lambda k: self.word_rank_idx[k], reverse=True)


    def summarize(self, sent_num=3):
        summary = []
        index = []

        for idx in self.sorted_sent_rank_idx[:sent_num]:
            index.append(idx)

        index.sort()

        for idx in index:
            summary.append(self.sentences[idx])
        return summary

    def keywords(self, word_num=10):
        rank = Rank()
        rank_idx = rank.get_ranks(self.words_graph)
        sorted_rank_idx = sorted(rank_idx, key=lambda x: rank_idx[x], reverse=True)

        keywords = []
        index = []
        for i in sorted_rank_idx[:word_num]:
            index.append(i)

        for i in index:
            keywords.append(self.idx2word[i])

        return keywords


# url = 'http://v.media.daum.net/v/20170611192209012?rcmd=r'
# textrank = TextRank(url)

# text = f"건강 이상설과 사망설이 제기됐던 김정은 북한 국무위원장의 공개 활동이 영상으로 공개됐다. 김 위원장은 보란듯이 자연스럽게 걸으며 담배를 피우는 등 여유있는 모습을 연출했다. 2일 오후 북한 조선중앙TV는 김 위원장이 전날 평안남도 순천 인비료공장 준공식에 참석한 소식을 약 15분 가량 보도했다. 공장 소개 영상 뒤 등장한 김 위원장은 건설자들의 '만세' 환호를 받으며 준공식 행사장으로 입장, 건재함을 과시했다. 김 위원장은 검은색 인민복 차림으로 자연스럽게 걸었으며, 계단 등을 내려갈 때도 부축을 받지 않았다. 지난달 11일 정치국회의 당시와 비교해 다소 피부가 타고 거칠어진 모습이긴 했지만 평소와 크게 달라지지 않은 외양이었다. 김 위원장은 연단에서 진지한 표정을 짓는 한편 맞은편에 앉은 김재룡 내각총리와 마주보면서 대화를 하고 크게 함박 웃음을 지어보이기도 했다."

text = ""
with open("article1.txt", mode='r', encoding='utf-8') as raw_text:
    text = ''.join(raw_text.readlines())

textrank = TextRank(text)

print("### 3줄 요약 ###")
for idx, row in enumerate(textrank.summarize(3)):
    print(f"{idx + 1} : {row}")
print('keywords :',textrank.keywords())

