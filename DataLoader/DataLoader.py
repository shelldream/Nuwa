#-*- coding:utf-8 -*-
"""
    Description:数据加载模块
        输入文件格式为按照特定分隔符分隔的符号序列，本模块将对整个原始训练数据构建词表，并构建模型的
        训练数据。每条训练样本是包含正负样本的三元组。
    Author: shelldream
    Date: 2018-07-28
"""
import sys
reload(sys).setdefaultencoding('utf-8')
import os
import logging
import random
import collections
from numpy import zeros, uint32

logging.basicConfig(level = logging.DEBUG,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class DataLoader(object):
    def __init__(self, filePath, vocPath, trainDataPath, 
            wcLimit=10, skipWin=5, skipNum=6, nsCnt=5, ns_exponent=0.75, sep="\t"):
        """
            Args:
                filePath: string, 原始训练数据文件名
                vocPath: string, 词表信息文件,每行包括词id、词、词频
                trainDataPath: string, 词向量训练数据，每行是一个包含正负样本的三元组
                wcLimit: int, 词表中词的出现的阈值，出现次数低于该值的低频词不会出现在词表中
                skipWin: int, 目标词最远可以联系的距离,词窗大小为2*skipWin+1
                skipNum: int, 一个词窗中生成正样本的词的个数
                nsCnt: int, 一条正样本对应的负样本数目
                ns_exponent: float, 生成负采样时的词频因子
                sep: string, 原始训练数据的分隔符
            Returns:

        """
        if not os.path.exists(filePath):
            raise ValueError("The file path %s is invalid! ", filePath)

        if skipNum > 2*skipWin:
            raise ValueError("skipNum cann't be larger than 2*skipWin")

        self.words = []
        self.filterWords = []
        self.wcDict = {}
        self.idx2w = {}
        self.w2idx = {}
        
        self.vocPath = vocPath
        self.trainDataPath = trainDataPath

        self.vocabSize = 0
        self.ns_exponent = ns_exponent
        self.wcLimit = wcLimit
        self.skipWin = skipWin
        self.skipNum = skipNum
        self.nsCnt = nsCnt
        self.sep = sep

        with open(filePath, "r") as fr:
            for line in fr:
                content = line.strip().split(self.sep)
                self.words.extend(content)
        self.__buildVocabulary()

        logging.info("Before filtering the word list size is %s", len(self.words))
        for word in self.words:  # 过滤低频词，并将原始的词转换成id
            if word in self.wcDict and word in self.w2idx:
                self.filterWords.append(self.w2idx[word])
        logging.info("After filtering the word list size is %s", len(self.filterWords))
        
        self.cumTable = zeros(self.vocabSize, dtype=uint32)
        self.__buildCumTable()
        self.__genTrainData()

    def __buildVocabulary(self):
        """根据原始训练数据构建词表，输出词表文件"""
        wcDict = {}
        for w in self.words:
            self.wcDict[w] = self.wcDict.get(w, 0) + 1

        for k, v in wcDict.iteritems():
            if v >= self.wcLimit:
                self.wcDict[k] = v

        sortedWcList = sorted(self.wcDict.iteritems(), key=lambda x:x[1], reverse=True)
        self.vocabSize = len(sortedWcList)
        logging.info("The vocabulary size is %d", self.vocabSize)
        idx = 0
        with open(self.vocPath, "w") as fw:
            for (w, c) in sortedWcList:
                self.idx2w[idx] = w
                self.w2idx[w] = idx
                fw.write(self.sep.join([str(idx), w, str(c)]))
                fw.write("\n")
                idx += 1

        logging.info("The buiding of vocabulary completed!")

    def __buildCumTable(self, domain=2**31-1):
        """
            参考资料：
                https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py#L1797
        """
        # compute sum of all power (Z in paper)
        train_words_pow = 0.0

        for wIdx in range(self.vocabSize):
            word = self.idx2w[wIdx]
            count = self.wcDict[word]
            train_words_pow += count ** self.ns_exponent

        cumulative = 0.0
        for wIdx in xrange(self.vocabSize):
            word = self.idx2w[wIdx]
            count = self.wcDict[word]
            cumulative += count**self.ns_exponent
            self.cumTable[wIdx] = round(cumulative / train_words_pow * domain)
        if len(self.cumTable) > 0:
            assert self.cumTable[-1] == domain

    def __negativeSampling(self, targetIdx):
        """
            Args:
                targetIdx: int, 正样本idx，返回的idx不能和它相同
            Returns:
                wIdx: int, 负采样噪生词的id
            
            https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py#L420
        """
        wIdx = targetIdx
        while wIdx == targetIdx:
            wIdx = self.cumTable.searchsorted(random.randint(0, self.cumTable[-1]))

        return wIdx

    def __genTrainData(self):
        """"""
        span = 2 * self.skipWin + 1
        if len(self.filterWords) < span:
            raise ValueError("The filterWords list is too short!!")
        
        logging.debug("The window size is %d", span)
        wordBuffer = collections.deque(maxlen=span)
        idx = 0
        # Initialize dequeue
        for idx in range(span-1):
            wordBuffer.append(self.filterWords[idx])

        # training data generation

        if os.path.exists(self.trainDataPath):
            cmd = "rm -rf {}".format(self.trainDataPath)
            print cmd
            os.popen(cmd)

        fa = open(self.trainDataPath, "a")
        for idx in range(span-1, len(self.filterWords)):
            wordBuffer.append(self.filterWords[idx])
            targetIdx = self.skipWin
            targetWIdx = wordBuffer[targetIdx]  # 中心词的idx
            invalidIdx = set()
            invalidIdx.add(targetIdx)
            for i in range(self.skipNum):
                posIdx = targetIdx
                while posIdx in invalidIdx:
                    posIdx = random.randint(0, span-1)
                invalidIdx.add(posIdx)
                posWIdx = wordBuffer[posIdx]
                for i in range(self.nsCnt):
                    negWIdx = posWIdx
                    while negWIdx == posWIdx:
                        negWIdx = self.__negativeSampling(targetWIdx)
                    fa.write(self.sep.join([self.idx2w[targetWIdx], self.idx2w[posWIdx], self.idx2w[negWIdx]]))
                    #fa.write(self.sep.join([str(targetWIdx), str(posWIdx), str(negWIdx)]))
                    fa.write("\n")  
        fa.close()


if __name__ == "__main__":
    dataLoader = DataLoader("test.txt", "testVoc.txt", "testTrain.txt", wcLimit=1, skipWin=3, skipNum=4, sep=" ")
