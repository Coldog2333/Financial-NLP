# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 09:51:06 2018

p.s. 计算senti-score之前要确保classifier已经初始化
     带模型参数的文件都请用空格分隔

@author: HP
"""

import os
from NLP import NLP
import pandas as pd
import multiprocessing

def is_cut_file(filename):
    return filename.split('.')[0][-3:]=='cut'
    
def remove_stopwords_in_valid_word(valid_word_filename):
    fp=open(NLP.stopwords_txt,'r',encoding='utf-8')
    stopwords=fp.readlines()
    fp.close()
    fp=open(valid_word_filename,'r',encoding='utf-8')
    valid_word=fp.readlines()
    fp.close()
    valid_word_new=[]
    for word in valid_word:
        if word in stopwords:
            continue
        else:
            valid_word_new.append(word)
    fp=open(valid_word_filename,'w',encoding='utf-8')
    for word in valid_word_new:
        fp.writelines(word)
    fp.close()
    
class Senti:
    common_word_filename=r'D:\Mainland\Campus Life\ZXtrainee\data\common_words.txt'
    article_dir='D:\\Mainland\\Campus Life\\ZXtrainee\\article'
    
    nlp_vector_filename=r'D:\Mainland\Campus Life\ZXtrainee\data\nlp_vector\common_words_vector 0_0.txt'
    valid_word_filename=r'D:\Mainland\Campus Life\ZXtrainee\data\valid_word_valid_words 0_0.txt' # word2vec中有的词
    word_score_filename=r'D:\Mainland\Campus Life\ZXtrainee\data\word_score\common_words_score 0_0.txt'
    
    w2v_mincount=nlp_vector_filename.split('_')[-2]
    w2v_size=nlp_vector_filename.split('_')[-1].split('.')[0]
    word_classifier=dict()
    
    def __init__(self, Min_count=0, Size=0):
        self.nlp=NLP()
        try:
            self.renew_model(Min_count, Size)
        except:
            pass
        
        
    def renew_model(self, Min_count, Size):
        suffix=str(Min_count) + '_' + str(Size) + '.txt'
        self.nlp_vector_filename=' '.join(self.nlp_vector_filename.split(' ')[:-1]) + ' ' + suffix
        self.valid_word_filename=' '.join(self.valid_word_filename.split(' ')[:-1]) + ' ' + suffix
        self.word_score_filename=' '.join(self.word_score_filename.split(' ')[:-1]) + ' ' + suffix
        self.renew_word_score(Min_count, Size)
    
    
    def renew_word_score(self, Min_count, Size):
        common_words=pd.read_csv(self.word_score_filename)
        self.word_classifier = dict(zip(common_words.iloc[:,0], common_words.iloc[:,1]))
        
        
    def get_topn_topm(self, s1, s2, n=10, m=3):
        s1_sorted=s1.sort_values(ascending=False)
        s1topn_index=s1_sorted.index[:n]
        d=dict()
        for i in s1topn_index:
            d[i[:-3]]=s2[i[:-3]+'wn']
        s=pd.Series(d)
        s_sorted=s.sort_values(ascending=False)
        l=len(s_sorted[s_sorted!=0])
        if(l==0):
            index=[]
            for i in range(m):
                index.append(s1topn_index[i][:-3])
            return index
        elif(l<m):
            return s_sorted.index[:l]
        else:
            return s_sorted.index[:m]            
            
    
    def score_of_common_words(self, Min_count, Size, saveflag=1, savefilename=''):
        """
        calculate scores of common words, and save the results as you like.
        
        p.s. please make sure you have set savefilename.
        """
        self.set_model_parameters(Min_count, Size)
        table=pd.read_csv(self.nlp_vector_filename)
        #table=table.abs() # 余弦相似度直接取绝对值
        result=['']*table.shape[0]
        score=[0]*table.shape[0]
        label_num=(table.shape[1]-1)/2
        for i in range(table.shape[0]): 
            w2v=table.iloc[i,1:label_num+1]
            wn=table.iloc[i,len(label_num)+1:len(label_num)*2+1]
            result[i]=self.get_topn_topm(w2v, wn, n=9, m=3) # 这是一个字符串Index
            for reword in result[i]:
                score[i]+=table.loc[i, reword+'w2v']*self.nlp.Label_dict[reword]
            score[i]/=len(result[i])
        
        if saveflag:
            try:
                fp=open(self.valid_word_filename,'r',encoding='utf-8')
                txtlist=fp.readlines()
            except:
                fp=open(self.valid_word_filename,'r',encoding='gbk')
                txtlist=fp.readlines()
            valid_words=[]
            for t in txtlist:
                t=t.split('\n')[0]
                valid_words.append(t)
            fp.close()
            rawdata=pd.DataFrame(score, valid_words)
            pd.DataFrame.to_csv(rawdata, savefilename,encoding='gkb')
    
    
    def score_of_article(self, article_filename, mode='text'):
        title=os.path.split(article_filename)[1].split('.')[0]
        try_cut_file=os.path.join(os.path.split(article_filename)[0],title)+'_cut.txt'
        if mode=='text':
            if os.path.exists(try_cut_file): # 如果cut file已经存在了就直接用cut file处理了
                words = self.nlp.txt2wordbag(try_cut_file, cutflag=False, remove_stopwords=False)
            else:
                words = self.nlp.txt2wordbag(article_filename, cutflag=True, remove_stopwords=True)
        elif mode=='title':
            words = self.nlp.title2wordbag(title, remove_stopwords=True)
        senti_score = 0
        count=0
        for i in words:
            x = self.word_classifier.get(i)
            if x == None:
                x = 0
                count += 1
            else:
                senti_score += x
        return senti_score/(len(words)-count), title
    
    
    def p_score_of_article(self, article_filename, mode):
        """
        info 用于返回值
        """
        title=os.path.split(article_filename)[1].split('.')[0]
        try_cut_file=os.path.join(os.path.split(article_filename)[0],title)+'_cut.txt'
        if mode=='text':
            if os.path.exists(try_cut_file): # 如果cut file已经存在了就直接用cut file处理了
                words = self.nlp.txt2wordbag(try_cut_file, cutflag=False, remove_stopwords=False)
            else:
                words = self.nlp.txt2wordbag(article_filename, cutflag=True, remove_stopwords=True)
        elif mode=='title':
            words = self.nlp.title2wordbag(title, remove_stopwords=True)
        senti_score_article = 0
        count=0
        for i in words:
            x = self.word_classifier.get(i)
            if x == None:
                x = 0
                count += 1
            else:
                senti_score_article += x
        #info.append((senti_score_article, title))
        #senti_score_date += senti_score_article
    
    
    def score_of_date(self, date='2018-08-01'):
        """
        Returns
            tuple: double   score_of_date
                   tuple    info:( title, score)
        """        
        senti_score=0
        articles = os.listdir(os.path.join(self.article_dir, date))
        info = []
        count = 0
        for article in articles:
            if is_cut_file(os.path.join(self.article_dir, date, article)):
                continue
            score, title = self.score_of_article(os.path.join(self.article_dir, date, article))
            senti_score += score
            info.append((title, score))
            count +=1
        return senti_score/len(articles), info
    
    
    def p_score_of_date(self, date='2018-08-01'):
        """
        It seems that it cannot work.
        Returns
            tuple: double   score_of_date
                   tuple    info:( title, score)
        """      
        #lock=threading.RLock()
        #senti_score_date=0
        articles = os.listdir(os.path.join(self.article_dir, date))
        #info=[]
        pool_arg = []
        count = 0
        p=multiprocessing.Pool(multiprocessing.cpu_count())
        for article in articles:
            if is_cut_file(os.path.join(self.article_dir, date, article)):
                continue
            pool_arg.append(os.path.join(self.article_dir, date, article))
            pool_arg.append('text')
            count += 1
        result = p.map(self.p_score_of_article, pool_arg)
        return result
        """for article in articles:
            if is_cut_file(os.path.join(self.article_dir, date, article)):
                continue
            t=threading.Thread(target=self.p_score_of_article, args=(os.path.join(self.article_dir, date, article), info))
            th.append(t)
            t.start()
            count += 1
        for t in th:
            t.join()
        for i in range(count):
            senti_score_date += info[i][0]
        return senti_score_date/count, info"""

    
    def calculate_scores_of_all(self, saveflag=0, savefilename=''):
        dates = os.listdir(self.article_dir)
        all_date_score=[]
        for date in dates:
            try:
                score,info=self.score_of_date(date)
                all_date_score.append((date,score))
            except:
                continue
        if saveflag:
            rawdata=pd.DataFrame(all_date_score)
            pd.DataFrame.to_csv(rawdata, savefilename)
        return all_date_score,dates
        
        
        
        
        
        