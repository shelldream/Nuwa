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
from collections import dequeue


class DataLoader(Object):
    def __init__(self, filePath, vocPath, trainDataPath, 
    		wcLimit=10, skipWin=5, skipNum=6, npRatio=3, sep="\t"):
    	"""
    		Args:
    			filePath: string, 原始训练数据文件名
    			vocPath: string, 词表信息文件,每行包括词id、词、词频
    			trainDataPath: string, 词向量训练数据，每行是一个包含正负样本的三元组
    			wcLimit: int, 词表中词的出现的阈值，出现次数低于该值的低频词不会出现在词表中
    			skipWin: int, 目标词最远可以联系的距离,词窗大小为2*skipWin+1
    			skipNum: int, 一个词窗中生成正样本的词的个数
    			npRatio: int, 负样本和正样本的比例及一条正样本对应的负样本数目
    			sep: string, 原始训练数据的分隔符
    		Returns:

    	"""
    	if os.path.exists(filePath):
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

    	self.wcLimit = wcLimit
    	self.skipWin = skipWin
    	self.skipNum = skipNum
    	
    	with open(filePath, "r") as fr:
    		for line in fr:
    			content = line.strip().split(sep)
    			self.words.extend(content)
    	self.__buildVocabulary()

    	logging.info("Before filtering the word list size is ", self.words.size())
    	for word in self.words:  # 过滤低频词，并将原始的词转换成id
    		if word in self.wcDict and word in self.w2idx:
    			self.filterWords.append(self.w2idx[word])
    	logging.info("After filtering the word list size is ", self.filterWords.size())
    	
    def __buildVocabulary(self):
    	"""根据原始训练数据构建词表，输出词表文件"""
    	wcDict = {}
    	for w in self.words:
    		wcDict[w] = self.wcDict.get(w, 0) + 1

    	for k, v in wcDict.iteritems():
    		if v >= self.wcLimit:
    			self.wcDict[k] = v

    	sortedWcList = sorted(self.wcDict.iteritems(), key=lambda x:x[1], reverse=True)
    	logging.info("The vocabulary size is ", sortedWcList.size())
    	idx = 0
    	with open(self.vocPath, "w") as fw:
    		for (w, c) in sortedWcList:
    			self.idx2w[idx] = w
    			self.w2idx[w] = idx
    			fw.write("%d\t%s\t%d\n", idx, w, c)
    			idx += 1

    	logging.info("The buiding of vocabulary completed!")

    def __wordSampling(self):
    	""""""
    	pass

    def __genTrainData(self):
    	""""""
    	span = 2 * self.skipWin + 1
    	if self.filterWords.size() < span:
    		raise ValueError("The filterWords list is too short!!")

    	wordBuffer = dequeue(maxlen=span)
    	idx = 0
    	# Initialize dequeue
    	for idx in range(span):
    		wordBuffer.append(self.filterWords[idx])

    	# training data generation
    	fa = open(self.trainDataPath, "a")
    	for idx in range(span, self.filterWords.size()):
    		targetIdx = self.skipWin
    		invalidIdx = set()
    		invalidIdx.add(targetIdx)
    		for i in range(self.skipNum):
    			while targetIdx in invalidIdx:
    				targetIdx = random.randint(0, span-1)
    			invalidIdx.add(targetIdx)

    	fa.close()



if __name__ == "__main__":
	dataLoader = DataLoader("test.txt")