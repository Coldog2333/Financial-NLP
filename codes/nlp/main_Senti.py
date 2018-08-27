# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 21:30:42 2018

@author: HP
"""

from Senti import Senti

s=Senti()
for i in range(3,6):
    for j in range(100,600,100):
        try:
            print('try to calculate senti-score of Min_count= %d, Size= %d' % (i,j))
            s.renew_model(Min_count=i, Size=j)
            s.calculate_scores_of_all(saveflag=1, 
                savefilename=r'D:\Mainland\Campus Life\ZXtrainee\data\date_score\data_score '+str(i)+'_'+str(j)+'.csv')
            print('successed')
        except:
            print('failed')
            continue