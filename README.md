***
# Constructing Financial Sentimental Factors in Chinese Market Using Techniques of Natural Language Processing

Introduction
====
Natural language processing, as one of the most promising fields of machine learning, has achieved great development recently and has been used in financial market. In this project, we are aiming to use an algotithm to analyze text data from influential financial websites to construct a sentimental factor which represents the daily sentiment of the market.  
And papers here: [***English version***](https://github.com/Coldog2333/Financial-NLP/tree/master/paper/Constructing%20Financial%20Sentimental%20Factors%20in%20Chinese%20Market%20Using%20Techniques%20of%20Natural%20Language%20Processing.pdf) and [***中文版***](https://github.com/Coldog2333/Financial-NLP/tree/master/paper/%E5%88%A9%E7%94%A8%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86%E6%8A%80%E6%9C%AF%E6%9E%84%E5%BB%BA%E4%B8%AD%E5%9B%BD%E5%B8%82%E5%9C%BA%E9%87%91%E8%9E%8D%E8%88%86%E6%83%85%E5%9B%A0%E5%AD%90.pdf).

Experiment
====
Correlation Between Sentimental Factor and Chinese Markets
-------
<p class="half" align="center">
  <img src="https://github.com/Coldog2333/Financial-NLP/blob/master/figure/Correlation_Between_Sentimental_Factor_and_SSE(en).png" width="350px" height="350px"/>
  <img src="https://github.com/Coldog2333/Financial-NLP/blob/master/figure/Correlation_Between_Sentimental_Factor_and_SZSE(en).png" width="350px" height="350px"/>
</p>


Time Series of Sentimental Factor and Chinese Markets
-------
+ As for SSE,

<p class="half" align="center">
   <img src="https://github.com/Coldog2333/Financial-NLP/blob/master/figure/senti-score_vs_SSE(en).png" width="800px" height="402px"/>
</p>

+ As for SZSE,

<p align="center">
   <img src="https://github.com/Coldog2333/Financial-NLP/blob/master/figure/senti-score_vs_SZSE(en).png" width="800px" height="402px"/>
</p>

Contribution
====
Contributors
-------
- ***Junfeng Jiang***
- ***Jiahao Li***

Institutions
-------
- ***AI&FintechLab of Likelihood Technology***
- ***Sun Yat-sen University***

Acknowledgement
-------
We would like to say thanks to MingWen Liu from ShiningMidas Private Fund for his generous help throughout the research. We are also grateful to Xingyu Fu from Sun Yatsen University for his guidance and help. With their help, this research has been completed successfully. 

Set up
====
Python Version
-------
- ***3.6***

Modules needed
-------
- ***os***
- ***six***
- ***codec***
- ***logging***
- ***jieba***
- ***gensim***
- ***nltk***
- ***selenium***
- ***numpy***
- ***pandas*** 
- ***threading***
- ***datetime***
- ***time***


Contact
====
- jiangjf6@mail2.sysu.edu.cn
- lijh76@mail2.sysu.edu.cn
- a412133593@gmail.com
