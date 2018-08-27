# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 09:42:56 2018

version v2.0
按日期分放文章

@author: HP

"""



import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from time import sleep
import datetime

#import win32api
work_space=r'D://Mainland/Campus Life/ZXtrainee/codes'
invalid_char=['|','.',':',',','*','\\','/','/']
#--------------------------------------------
def sdtime(time):
    # 改year-month-day 为标准时间格式：year年month月day日
    part=time.split('-')
    if len(part)==2: # 没有year
        year=str(datetime.datetime.now().year)
    else:
        year=part[-3]
    month=part[-2]
    day=part[-1]
    return year+'-'+month+'-'+day


def clean_invalid(string):
    flag=0
    for c in invalid_char:
        if c in string:
            string=string.replace(c,' ')
            flag=1
    return string,flag
    
def load(x,article_number):
    flag=1 # 表示尚未拉到底部
    L=0
    while(flag or L<article_number):
        L=len(x.find_elements_by_class_name('home__timeline__item'))
        try:
            x.find_element_by_class_name('home__timeline__more').click()
            flag=0
        except:
            x.find_elements_by_class_name('home__timeline__item')[-1].find_element_by_xpath('h3/a').send_keys(Keys.TAB)
    print('loaded %d articles.' % L)

if not os.path.exists(work_space+'/article'):
    os.makedirs(work_space+'/article')  
log_file=work_space+'/article/log.txt'

def download(x):
    count=0
    article_link=x.find_elements_by_class_name('home__timeline__item')
    logfp=open(log_file,'a')
    for l in article_link:
        try:
            time=l.find_element_by_class_name('timestamp').text.split(' ')[0]
            time=sdtime(time)
            l.find_element_by_xpath('h3/a').click()
            t=l.find_element_by_xpath('h3/a').text
            title,flag=clean_invalid(t)
            if flag:
                logfp.writelines(t)
                logfp.write('\n')
            windows = x.window_handles
            x.switch_to.window(windows[-1])
            txt=x.find_element_by_xpath('//div[@class="article__bd__detail"]').text
                
            if not os.path.exists(work_space+'/article/'+time+'/'):
                os.makedirs(work_space+'/article/'+time+'/')
            txt_filename=work_space+'/article/'+time+'/'+title+'.txt'
            fp=open(txt_filename,'w',encoding='utf-8')
            fp.writelines(txt)
            fp.close()
            x.close()
            count+=1
            x.switch_to.window(windows[0])
        except:
            windows = x.window_handles
            x.switch_to.window(windows[0])
            continue
    logfp.close()
    print('downloaded %d articles.' % count)

prev_time = datetime.datetime.now() #当前时间        
url_xueqiu=r'https://xueqiu.com/'
x=webdriver.Chrome(r'C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe')
x.get(url_xueqiu)
load(x,1000)
download(x)
x.quit()
cur_time = datetime.datetime.now()  #训练后此时时间
h, remainder = divmod((cur_time - prev_time).seconds, 3600)
m, s = divmod(remainder, 60)
print("It costs %02d:%02d:%02d" % (h, m, s))