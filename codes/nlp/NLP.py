# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 16:43:20 2018

version: v0.5

p.s: 1. 用之前修改一下停用词库stopwords的文本路径
     2. 用之前修改一下中文wordnet数据库zh_wordnet的文本路径
     3. 用之前修改一下标签词集label_dict的文本路径
    
bug: 1. similarity出现了负数
     2. cut大文件的时候好像会让电脑崩溃，不知道是我的电脑问题还是怎么样。

@author: HP
"""

import os
import jieba
import logging
import gensim
import codecs
from six import string_types
import numpy as np
import pandas as pd
from gensim.models import word2vec
from gensim import matutils
from nltk.corpus import wordnet as wn
import datetime

zh_symbol=['！','？','，','。','【','】','（','）','￥','…','—','《','》','”','“','：','；','、','‘','’']
number=['0','1','2','3','4','5','6','7','8','9']
letter=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
        'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']    

# NLP 工具类
class NLP:
    stopwords_txt=r'D:\Mainland\Campus Life\ZXtrainee\data\stopwords'
    wordnet_txt=r'D:\Mainland\Campus Life\ZXtrainee\data\cmn\cow\cow-not-full.txt'
    label_file=r'D:\Mainland\Campus Life\ZXtrainee\data\label.csv'
    process_file=''
    #save_model_name=''
    model=None
    len_vector=0

    def __init__(self):
        self.loadWordNet() 
        self.loadstopwords()
        self.load_label()
        
        self.Label_wn=dict()
        for key in self.Label_dict.keys():
            ll=self.findWordNet(key)
            self.Label_wn[key]=list()
            for l in ll:
                self.Label_wn[key].append(self.id2ss(l))
                
        self.Label_vec=None
        self.Label_vec_u=None
        
        
    def loadWordNet(self):
        """
        load zh_wordnet into the object.
        将cow-not-full文件中的数据集整合成set
        """
        f = codecs.open(self.wordnet_txt, "rb", "utf-8")
        self.known = dict()
        #self.known = set()
        for l in f:
            if l.startswith('\ufeff#') or not l.strip():
                continue
            row = l.strip().split("\t")
            (synset,lemma)=row
            #if len(row) == 2:
            #    (synset, lemma) = row 
            #elif len(row) == 3:
            #    (synset, lemma, status) = row #根本就没有三个东西的项
            #else:
            #    print("illformed line: ", l.strip())
            #if not (synset.strip(), lemma.strip()) in self.known:
            #    self.known.add((synset.strip(), lemma.strip()))
            if not lemma.strip() in self.known.keys():
                self.known[lemma.strip()]=[]
            self.known[lemma.strip()].append(synset)
    
    
    def loadstopwords(self):
        """
        load stopwords into the object.
        """
        self.stop_words=list()
        stop_f=open(self.stopwords_txt,'r',encoding='utf-8')
        for line in stop_f.readlines():
            line=line.strip()
            if not len(line):
                continue
            self.stop_words.append(line)
        stop_f.close()
 
      
    def load_label(self):
        """
        load label dictionary into the object.
        the format must be like this:
            积极,消极
            p1,n1
            p2,n2
            ...,...
            pk,nk
        """
        table=pd.read_csv(self.label_file)
        pos=table.loc[:,'积极'].tolist()
        neg=table.loc[:,'消极'].tolist()
        self.Label_index=pos+neg
        self.Label_dict=dict(zip(pos,[1]*len(pos)))
        self.Label_dict.update(dict(zip(neg,[-1]*len(neg))))
        
        
    def cut(self, origin_file, cut_file='', remove_stopwords=False, swith_to_newtxt=False, delflag= True):
        """
        Parameters
            ----------
            origin_file : str
                name(absolute path) of original text file.
            cut_file : str
                where cut text file saves.
                default: in the same place with original text file, and named file_cut
            remove_stopwords : bool
                remove the stopwords when cutting?
            swith_to_newtxt : bool
                want to swith to cut_file after cutting? if yes, it will return the name of cut_file.
            delflag : bool
                del all the symbols except chinese words?
                训练时就不删掉吧？（delflag= False吧？）
        Returns
            if swith_to_newtxt==False : Nothing
            if swith_to_newtxt==True  : name of cut_file
            
        Tips:
            When cutting for training, I advice you should set
                + remove_stopwords = False
                + switch_to_newtxt = True
                + delflag=False
            When cutting a raw new text, I advice you should set
                + remove_stopwords = True
                + delflag=True
        """
        print('start cutting...\n')
        prev_time = datetime.datetime.now() #当前时间  
        
        if cut_file=='':    # 若没有指定cut file文件名则默认放在original file同一位置下。
            cut_file=origin_file.split('.')[0]+'_cut.'+origin_file.split('.')[1]
        # 兼容utf-8和gbk编码
        try:
            fp=open(origin_file,'r',encoding='utf-8')
            raw_text=fp.read()      #text without cutting
            code='utf-8'
        except:
            fp=open(origin_file,'r',encoding='gbk')
            raw_text=fp.read()      #text without cutting
            code='gbk'
        fp.close()
        
        words = jieba.cut(raw_text, cut_all=False, HMM=True)
        str_cut=' '.join(words) # 未经任何处理的分词后字符串
        
        if delflag is True:
            for sym in zh_symbol: # remove chinese symbols
                str_cut=str_cut.replace(sym,'')
            for sym in number:    # remove number
                str_cut=str_cut.replace(sym,'')
            for sym in letter:    # remove english letter
                str_cut=str_cut.replace(sym,'')
            strlist_cut=str_cut.split(' ')
            """for string in strlist_cut: # remove english letter
                try:
                    if (string[0]>='A' and string[0]<='Z') or(string[0]>='a' and string[0]<='z'):
                        strlist_cut.remove(string)
                except:
                    continue"""
        
        if(remove_stopwords==True):
            strlist_new=[]
            for word in strlist_cut:
                if (not len(word)) or (word in self.stop_words):
                    continue
                else:
                    strlist_new.append(word)
            str_cut=' '.join(strlist_new)
        # 兼容处理，若用户还没有创建cut文件夹就开始创建文件
        try:
            fp=open(cut_file,'w',encoding=code)
        except Exception as e:
            if(type(e)==FileNotFoundError):
                os.mkdir(os.path.split(cut_file)[0])
                fp=open(cut_file,'w',encoding=code)
            else:
                raise e
        fp.writelines(str_cut)  # 将分词后文本数据写入文件
        fp.close()
        
        cur_time = datetime.datetime.now()  #分词后此时时间
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)    
        print('done.') 
        print("It costs %02d:%02d:%02d to cut." % (h, m, s))
        
        if swith_to_newtxt:
            return cut_file
        

    def txt2sentence(self, filename):    
        """
        read a <cut_file> and return an iterator sentences
        (that is a list of some lists, and the second 'list' is a list of words ).
        """
        sentences=[]
        try:
            fp=open(filename,'r',encoding='utf-8')
            lines=fp.readlines()
        except:
            fp=open(filename,'r',encoding='gbk')
            lines=fp.readlines()

        for line in lines:
            line = line.strip()
            if len(line)<=1:
                continue
            line=line.replace('\n','').replace('\r','').split(' ')
            sentences.append(line)
        return sentences

    
    def safe_renew_label_vec(self):
        """
        initialize word vectors of words in label_dict.
        origin version(safe)
        """
        self.Label_vec=np.empty((len(self.Label_dict),self.len_vector))
        self.Label_vec_u=np.empty((len(self.Label_dict),self.len_vector))
        for i in range(len(self.Label_index)):
            try:
                self.Label_vec[i,:]=self.model.wv.__getitem__(self.Label_index[i])
                self.Label_vec_u[i,:]=matutils.unitvec(self.model.wv.__getitem__(self.Label_index[i]))
            except:
                self.Label_vec[i,:]=np.zeros((1,self.len_vector)) # debug期间先这样处理吧
                self.Label_vec_u[i,:]=np.zeros((1,self.len_vector))
    
    
    def renew_label_vec(self):
        """
        initialize word vectors of words in label_dict.
        fast version(unstable)
        !Attention! : use it only when you make sure that all words in Label_index can calculate the word vector.
        """
        self.Label_vec=self.model.wv.__getitem__(self.Label_index)
        self.Label_vec_u=unitvec(self.Label_vec)
        
    
    def renew_label_wn(self): # maybe it will never be used, because our zh_wordnet maybe never update.
        for key in self.Label_dict.keys():
            self.Label_wn[key]=self.id2ss(key)
    
    
    def save_model(self, save_model_name):
        """
        save model as save_model_name
        """
        self.model.save(save_model_name)
        # self.len_vector=self.model.trainables.layer1_size
        try:
            self.renew_label_vec()
        except:
            self.safe_renew_label_vec()
                
    
    def load_model(self, save_model_name):
        """
        load model into the object(self.model)
        """
        self.model=word2vec.Word2Vec.load(save_model_name)
        self.len_vector=self.model.trainables.layer1_size
        try:
            self.renew_label_vec()
        except:
            self.safe_renew_label_vec()
            
    
    def train_Word2Vec(self, train_corpus, saveflag=False, save_model_name='NLP_model', Size=100, Min_count=5):#, show_process=True):
        """
        train the word2vec model with the processing file.
        Parameters
            ----------
            train_corpus : str/list of lists
                name(absolute path) of train_corpus.
                of a list of sentences(a sentence is a list of words).
            saveflag : bool
                save trained model locally?
            save_model_name : str
                the model name(absolute path)
                default: 'NLP_model'
            Size : int
                length of the word vector
            Min_count : int
                minimum frequence can a word record on dictionary.
        Returns
            Nothing
        """
        print('start training...')
        prev_time = datetime.datetime.now() #当前时间    
        
        self.len_vector=Size
        #if show_process==True:
        #    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)   
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  
        if isinstance(train_corpus, string_types):
            sentences=self.txt2sentence(train_corpus)
        else:    
            sentences=train_corpus
        self.model=gensim.models.Word2Vec(sentences, size=Size, min_count=Min_count) #word to vector\in R^Size
        if saveflag:
            self.save_model(save_model_name) # save model locally
        try:
            self.renew_label_vec()
        except:
            self.safe_renew_label_vec()
        
        cur_time = datetime.datetime.now()  #训练后此时时间
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        print('done.')
        print("It costs %02d:%02d:%02d to train word2vec model." % (h, m, s))
        # model.wv.save_word2vec_format(save_model_name+".bin",binary=True)
        
    
    def show_Word2Vec(self, s, k=1, mode='topk'):
        """
        not often use now.
        Parameters
            ----------
            save_model_name : str
                the name of saved model
            s : str
            k : int/str
                if mode='similarity', it's a string.
                if mode='topk', it's a number, and defaultly 1.
            mode : str
                'similarity' : calculate the similarity between s and k, and note that k is a string.
                'topk' (default): find top k similar words of s, and note that k is a integer.
        Returns
            ----------
            float
                if mode='similarity', this is the similarity between s and k.
                if mode='return_topk', it'll not return a number but a iterator.
                if mode='topk', it'll print the most similar k words.
        """
        if self.model is None:
            raise Exception("no model")
            #model=word2vec.Word2Vec.load(save_model_name)
        if mode=='topk':
            y=self.model.most_similar(s,topn=k)
            print('与"%s"最相关的词有:\n' % s)
            for item in y:
                print(item[0],item[1])
        
        elif mode=='return_topk':
            return self.model.wv.most_similar(s,topn=k)
            #return model.most_similar(s,topn=k)
                
        elif mode=='similarity':
            y=self.model.wv.similarity(s,k) 
            # 余弦相似度，即对于两个向量v1,v2，先单位化后，再求内积。
            print('"%s"和"%s"的相似度为：%f%%' % (s,k,(y*100)))
            return y
        
        elif mode=='vector':
            print(self.model[s])
    
    
    def id2ss(self,ID):
        """
        Parameters
            ----------
            ID : str
                the id of a chinese word found in zh_wordnet.
        Returns
            ----------
            nltk.corpus.reader.wordnet.Synset
                an object in en_wordnet.
        """
        return wn.synset_from_pos_and_offset(str(ID[-1:]), int(ID[:8]))   
    
    
    def set_process_file(self, filename):
        """
        will be removed.
        set the processing filename.
        """
        self.process_file=filename
    
    
    def findWordNet(self, key):
        """
        Parameters
            ----------
            key : str
                the word you want to find in zh_wordnet.
        Returns
            ----------
            list
                a list in wordnet format.
        """
        try:
            return self.known[key]
        except:
            return list()
        """for kk in self.known:
            if (kk[1] == key):
                 ll.append(kk[0])"""
    
    
    def get_wnsimilarity(self, s0, s1):
        """
        not often use.
        Parameters
            ----------
            s0 : str 
                the first word
            s1 : str
                the second word
        Returns
            ----------
            float
                the path similarity in wordnet between s0 and s1.
        """
        l0 = self.findWordNet(s0)
        l1 = self.findWordNet(s1)
        if (len(l1)==0) or (len(l0)==0):
            return 0
        else:
            similarity=0
            count=0
            for wid0 in l0:
                desc0=self.id2ss(wid0)
                for wid1 in l1:
                    desc1=self.id2ss(wid1)
                    sim=desc0.path_similarity(desc1)
                    if(sim!=None):
                        similarity+=desc0.path_similarity(desc1)
                    else:
                        count+=1
            try:
                similarity/=(len(l0)*len(l1)-count) # 或者可以取最大的k个用于计算平均的similarity
            except:
                similarity=0
        return similarity


    def txt2wordbag(self, origin_file, cutflag=False, remove_stopwords=True): #testing
        """
        please remember to set a corresponding processing file.
        """
        if origin_file.split('.')[0][-3:]!='cut':
            cut_file=self.cut(origin_file, remove_stopwords=True, swith_to_newtxt=True)
        else:
            cut_file=origin_file
    
        try:
            fp=open(cut_file,'r',encoding='utf-8')
            rawtxt=fp.read()
        except:
            fp=open(cut_file,'r',encoding='gbk')
            rawtxt=fp.read()
        words_list=rawtxt.split(' ')
        new_words_list=[]
        for word in words_list:
            if word=='' or (ord(word[0])<1024):
                continue
            else:
                new_words_list.append(word)
        if new_words_list=='\u3000':
            return new_words_list[1:]
        else:
            return new_words_list
    
    
    def title2wordbag(self, title, remove_stopwords=True):
        words=jieba.cut(title,cut_all=False)
        str_cut=' '.join(words)
        for sym in zh_symbol: # remove chinese symbols
            str_cut=str_cut.replace(sym,'')
        for sym in number:    # remove number
            str_cut=str_cut.replace(sym,'')    
        strlist_cut=str_cut.split(' ')
        
        strlist_new=[]
        for word in strlist_cut: # remove english letter
            if (not len(word)) or (word in self.stop_words):
                continue
            elif (word[0]>='A' and word[0]<='Z') or(word[0]>='a' and word[0]<='z'):
                continue
            elif(ord(word[0])<1024):
                continue
            else:
                strlist_new.append(word)
        return strlist_new
    
    
################################ problem #####################################
    def wordbag2mat(self, wordbag): #testing
        if self.model==None:
            raise Exception("no model")
        matrix=np.empty((len(wordbag),self.len_vector))
        #如果词典中不存在该词，抛出异常，但暂时还没有自定义词典的办法，所以暂时不那么严格
        #try:
        #    for i in range(len(wordbag)):
        #        matrix[i,:]=self.model[wordbag[i]]
        #except:
        #    raise Exception("'%s' can not be found in dictionary." % wordbag[i])
        #如果词典中不存在该词，则push进一列零向量
        for i in range(len(wordbag)):
            try:
                matrix[i,:]=self.model.wv.__getitem__(wordbag[i])#[wordbag[i]]
            except:
                matrix[i,:]=np.zeros((1,self.len_vector))
        return matrix
################################ problem #####################################

    
    def similarity_label(self, words, normalization=True):
        """
        you can calculate more than one word at the same time.
        """
        if self.model==None:
            raise Exception('no model.')
        if isinstance(words, string_types):
            words=[words]
        vectors=np.transpose(self.model.wv.__getitem__(words))
        if normalization:
            unit_vector=unitvec(vectors,ax=0) # 这样写比原来那样速度提升一倍
            #unit_vector=np.zeros((len(vectors),len(words)))
            #for i in range(len(words)):
            #    unit_vector[:,i]=matutils.unitvec(vectors[:,i])
            dists=np.dot(self.Label_vec_u, unit_vector)
        else:
            dists=np.dot(self.Label_vec, vectors)
        return dists
    
    
    def topn_similarity_label(self, words, topn=10, normalization=True):
        if self.model==None:
            raise Exception('no model.')
        if isinstance(words, string_types):
            words=[words]
        
            """ we can discard this version.
            vectors=np.transpose(self.model.wv.__getitem__(words))
            if normalization:
                unit_vector=np.zeros((len(vectors),len(words)))
                for i in range(len(words)):
                    unit_vector[:,i]=matutils.unitvec(vectors[:,i])
                dists=np.dot(self.Label_vec_u, unit_vector)
            else:
                dists=np.dot(self.Label_vec, vectors)
            # 排除掉自身（因为有可能word本身就在label_dict里）
            # best = matutils.argsort(dists, topn = topn+1, reverse=True)
            # result = [(self.index2word[sim], float(dists[sim])) for sim in best if sim not in all_words]
            best = matutils.argsort(dists[:,0], topn = topn, reverse=True)
            result = [(self.Label_index[sim], float(dists[sim])) for sim in best]
            return result
        else:
            """
        vectors=np.transpose(self.model.wv.__getitem__(words))
        if normalization:
            unit_vector=unitvec(vectors,ax=0)
            dists=np.dot(self.Label_vec_u, unit_vector)
        else:
            dists=np.dot(self.Label_vec, vectors)
            #topwords=np.empty((topn,len(words)), np.string_)
        topwords=[]
        topsims=np.empty((topn,len(words)))
        best = np.argsort(dists, axis=0)
        for i in range(topn):
            topword=[]
            for j in range(len(words)):
                topword.append(self.Label_index[best[-i-1][j]])
                topsims[i][j]=dists[best[-i-1][j]][j]
            topwords.append(topword)
        result=[(topwords[i], topsims[i]) for i in range(topn)]
        return result
        """ print this result by:

            | for iword,isim in result:  |
            |     print(iword, isim)     |
            or
            | for iword, isim in b:                               |
            |     for i in range(len(b[0])):                      |
            |         print("%s:%f\t" %(iword[i],isim[i]),end="") |
            |     print("")                                       |
                
        """
                
    def synonym_label(self, word, calc='all' ,calc_k=5):
        """
        you can only calculate one word for one time.
        """
        ww=list()
        for w in self.findWordNet(word):
            ww.append(self.id2ss(w))
        if (len(ww)==0):
            #return 0
            raise Exception('cannot be found in zh_wordnet.')
        else:
            similarities=[0]*len(self.Label_index)
            if calc=='all': # 默认全部平均
                for i in range(len(self.Label_index)):
                    count=0
                    for w in ww:
                        for l in self.Label_wn[self.Label_index[i]]:
                            sim=w.path_similarity(l)
                            if(sim!=None):
                                similarities[i]+=sim
                            else:
                                count+=1
                    try:
                        similarities[i]/=(len(ww)*len(self.Label_wn[self.Label_index[i]])-count) # 平均similarity
                    except:
                        similarities[i]=0
                        
            elif calc=='calc_k': # 仅取前calc_k个词义
                for i in range(len(self.Label_index)):
                    count=0
                    simlist=[]
                    for w in ww:
                        for l in self.Label_wn[self.Label_index[i]]:
                            sim=w.path_similarity(l)
                            if(sim!=None):
                                simlist.append(sim)
                                count+=1
                    if count<=calc_k:
                        similarities[i]=np.mean(simlist)
                    else:
                        simlist=sorted(simlist,reverse=True)
                        similarities[i]=simlist[:calc_k-1]/calc_k # 取最大的k个用于计算平均的similarity
        return np.array(similarities)


    def topn_synonym_label(self, word, topn=10, calc='all', calc_k=5):
        ww=list()
        for w in self.findWordNet(word):
            ww.append(self.id2ss(w))
        if (len(ww)==0):
            return 0
        else:
            similarities=[0]*len(self.Label_index)
            if calc=='all': # 默认全部平均
                for i in range(len(self.Label_index)):
                    count=0
                    for w in ww:
                        for l in self.Label_wn[self.Label_index[i]]:
                            sim=w.path_similarity(l)
                            if(sim!=None):
                                similarities[i]+=sim
                            else:
                                count+=1
                    try:
                        similarities[i]/=(len(ww)*len(self.Label_wn[self.Label_index[i]])-count) # 平均similarity
                    except:
                        similarities[i]=0
                        
            elif calc=='calc_k': # 仅取前calc_k个词义
                for i in range(len(self.Label_index)):
                    count=0
                    simlist=[]
                    for w in ww:
                        for l in self.Label_wn[self.Label_index[i]]:
                            sim=w.path_similarity(l)
                            if(sim!=None):
                                simlist.append(sim)
                                count+=1
                    if count<=calc_k:
                        similarities[i]=np.mean(simlist)
                    else:
                        simlist=sorted(simlist,reverse=True)
                        similarities[i]=simlist[:calc_k-1]/calc_k # 取最大的k个用于计算平均的similarity
                        
        best=matutils.argsort(similarities, topn = topn, reverse=True)
        result = [(self.Label_index[sim], float(similarities[sim])) for sim in best]
        return result
      
    def safe_nlp_vector(self, words):
        """
        Parameters
            ----------
            words : list of str/str 
                wordbag
        Returns
            ----------
            ndarray(float)
                the corresponding vectors of words in wordbag.
                a vector contains the similarities calculated by word2vec and wordnet.
        """
        if isinstance(words, string_types):
            synonym=self.synonym_label(words)
            similarity=self.similarity_label(words)
        else:
            synonym=np.empty((len(self.Label_index),len(words)))
            similarity=np.empty((len(self.Label_index),len(words)))
            for i in range(len(words)):
                try:
                    synonym[:,i]=self.synonym_label(words[i])
                except:
                    synonym[:,i]=np.zeros((len(self.Label_index),1))[:,0]
                try:    
                    similarity[:,i]=self.similarity_label(words[i])[:,0]
                except:
                    similarity[:,i]=np.zeros((len(self.Label_index),1))[:,0]
        vector=np.concatenate((similarity, synonym))
        return vector
    
    
    def nlp_vector(self, words):
        if isinstance(words, string_types):
            synonym=self.synonym_label(words)
            similarity=self.similarity_label(words)
        else:
            synonym=self.synonym_label(words)
            similarity=np.empty((len(self.Label_index),len(words)))
            for i in range(len(words)):
                try:
                    similarity[:,i]=self.similarity_label(words[i])[:,0]
                except:
                    similarity[:,i]=np.zeros((len(self.Label_index),1))[:,0]
        vector=np.concatenate((similarity, synonym))
        return vector
    
    def example(self):
        """
        from NLP import NLP
        nlp=NLP()
        
        train:
        
        
        calculate the nlp_vector:
            nlp.load_model(model_name)
            try:
                # words=['上涨','下跌']
                vector=nlp.nlp_vector(words)
            except:
                vector=nlp.safe_nlp_vector(words)
        """
        
        
   # def judge8word(self, word, ):
        
    #def judge8article(self, article,):
     #   pass
#---------------------------------math----------------------------------------
     
def unitvec(vector, ax=1):
    v=vector*vector
    if len(vector.shape)==1:
        sqrtv=np.sqrt(np.sum(v))
    elif len(vector.shape)==2:
        sqrtv=np.sqrt([np.sum(v, axis=ax)])
    else:
        raise Exception('It\'s too large.')
    if ax==1:
        result=np.divide(vector,sqrtv.T)
    elif ax==0:
        result=np.divide(vector,sqrtv)
    return result

     