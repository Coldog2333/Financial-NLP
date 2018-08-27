# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 10:12:56 2018

@author: HP
"""

from NLP import NLP
import pandas as pd

common_words_filename=r'D:\Mainland\Campus Life\ZXtrainee\data\common_words.txt'
model_place='D:\\Mainland\\Campus Life\\ZXtrainee\\data\\zhwiki_latest\\model\\'
save_result_place='D:\\Mainland\\Campus Life\\ZXtrainee\\data\\'
min_count=3
size=100
saveflag=0

def calculate_nlpv(min_count,size,saveflag):
    nlp=NLP()
    nlp.load_model(model_place+str(min_count)+'_'+str(size)+'\\wiki_nlp_'+str(min_count)+'_'+str(size)+'.model')
    words=[]
    fp=open(common_words_filename,'r',encoding='utf-8')
    for line in fp.readlines():
        try:
            temp=nlp.model.wv.__getitem__(line.strip('\n')) # 检查词是否在word2vec模型中
            words.append(line.strip('\n'))
        except:
            continue
    try:
        vector=nlp.nlp_vector(words)
    except:
        vector=nlp.safe_nlp_vector(words)
    
    
    if saveflag:
        vector_transform = vector.T
        names=[]
        for index in nlp.Label_index:
            names.append(index+'w2v')
        for index in nlp.Label_index:
            names.append(index+'wn')
        text = pd.DataFrame(columns = names, data = vector_transform)
        #生成的csv文件的地址
        text.to_csv(save_result_place+'common_words_vector_'+str(min_count)+'_'+str(size)+'.csv')
        

calculate_nlpv(min_count, size, saveflag=1)
calculate_nlpv(3,200, saveflag=1)
calculate_nlpv(3,300, saveflag=1)
calculate_nlpv(4,500, saveflag=1)
calculate_nlpv(5,100, saveflag=1)
calculate_nlpv(5,200, saveflag=1)
calculate_nlpv(5,300, saveflag=1)
calculate_nlpv(5,400, saveflag=1)
