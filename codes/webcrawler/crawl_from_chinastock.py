# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 09:55:26 2018

@author: HP
"""

import os
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from time import sleep
import datetime
import threading
import multiprocessing

work_space=r'D:/Mainland/Campus Life/ZXtrainee'
invalid_char=['|','.',':',',','*','\\','/','/','?','"']

def sdtime(time):
    time_part=time.split('/')
    if(len(time_part)==2):
        year=str(datetime.datetime.now().year)
    else:
        year=time_part[-3]
    month=time_part[-2]
    day=time_part[-1]
    return year+'-'+month+'-'+day

def clean_invalid(string):
    flag=0
    for c in invalid_char:
        if c in string:
            string=string.replace(c,' ')
            flag=1
    return string,flag
    
decode=dict()
decode['宏观经济研究']=['%E5%AE%8F%E8%A7%82%E7%BB%8F%E6%B5%8E%E7%A0%94%E7%A9%B6','ca01']
decode['国内财经']    =['%E5%9B%BD%E5%86%85%E8%B4%A2%E7%BB%8F','ca03']
decode['要闻']        =['%E8%A6%81%E9%97%BB','ca01']
decode['股市要闻']    =['%E8%82%A1%E5%B8%82%E8%A6%81%E9%97%BB','ca01']
decode['盘面追踪']    =['%E7%9B%98%E9%9D%A2%E8%BF%BD%E8%B8%AA','ca01']
decode['大盘']        =['%E5%A4%A7%E7%9B%98%E5%88%86%E6%9E%90','ca04']
decode['行业动态']    =['%E8%A1%8C%E4%B8%9A%E5%8A%A8%E6%80%81','ca01']
decode['行业研究']    =['%E8%A1%8C%E4%B8%9A%E7%A0%94%E7%A9%B6','ca02']

def goto_npage(x, section, n=1):
    """
    defaultly turn to the first page.
    """
    url_format='http://www.chinastock.com.cn/information.do?methodCall=newsList&id=zxzx_'+decode[section][0]+'&pageNum='+str(n)+'&pageSize=20'
    x.get(url_format)


if not os.path.exists(work_space+'/article'):
    os.makedirs(work_space+'/article')
log_file=work_space+'/article/log.txt'


def download(*arg):
    try:
        x=webdriver.Chrome(r'C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe')
        section=arg[0]
        count=0
        logfp=open(log_file,'a')
        #download(section, k)
        if len(arg)==2:
            k=arg[1]
            goto_npage(x,section)
            page=1
            while(count<k):
                try:
                    article_links=x.find_element_by_id(decode[section][1]).find_elements_by_xpath('ul/li')
                except:
                    break
                for l in article_links:
                    try:
                        title=l.find_element_by_class_name('L').text
                        l.find_element_by_xpath('span/a').click()
                        windows=x.window_handles
                        x.switch_to_window(windows[-1])
                        txt=x.find_element_by_class_name('d_content').text
                        time=x.find_element_by_class_name('d_date').text
                        if time.split(' ')[0].split('\u3000')[-1][0]>='0' and time.split(' ')[0].split('\u3000')[-1]<='9':
                            time=time.split(' ')[0].split('\u3000')[-1]
                        else:
                            time=time.split(' ')[1].split('\u3000')[-1]
                        #time=sdtime(time)
                        title,flag=clean_invalid(title)
                        if flag:
                            logfp.writelines(title)
                            logfp.write('\n') 
                        if not os.path.exists(work_space+'/article/'+time):
                            os.makedirs(work_space+'/article/'+time)
                        txt_filename=work_space+'/article/'+time+'/'+title+'.txt'
                        fp=open(txt_filename,'w',encoding='utf-8')
                        fp.writelines(txt)
                        fp.close()
                        x.close() 
                        count+=1
                        x.switch_to.window(windows[0])
                    except Exception as e:
                        x.switch_to.window(x.window_handles[0])
                        continue
                #x.find_element_by_class_name('pagination_next').find_element_by_xpath('a').click()
                page+=1
                try:
                    goto_npage(x, section, page)
                except:
                    try:
                        sleep(1)
                        goto_npage(x, section, page)
                    except:
                        break 
            
        #download(section, From, To)
        elif len(arg)==3:
            From=arg[1]
            To=arg[2]
            goto_npage(x,section,From)
            page=From
            count=0
            logfp=open(log_file,'a')
            for i in range(From, To+1):
                try:
                    article_links=x.find_element_by_id(decode[section][1]).find_elements_by_xpath('ul/li')
                except:
                    break
                for l in article_links:
                    try:
                        title=l.find_element_by_class_name('L').text
                        l.find_element_by_xpath('span/a').click()
                        windows=x.window_handles
                        x.switch_to_window(windows[-1])
                        txt=x.find_element_by_class_name('d_content').text
                        time=x.find_element_by_class_name('d_date').text
                        if time.split(' ')[0].split('\u3000')[-1][0]>='0' and time.split(' ')[0].split('\u3000')[-1]<='9':
                            time=time.split(' ')[0].split('\u3000')[-1]
                        else:
                            time=time.split(' ')[1].split('\u3000')[-1]
                        #time=sdtime(time)
                        title,flag=clean_invalid(title)
                        if flag:
                            logfp.writelines(title)
                            logfp.write('\n')
                        if not os.path.exists(work_space+'/article/'+time):
                            os.makedirs(work_space+'/article/'+time)
                        txt_filename=work_space+'/article/'+time+'/'+title+'.txt'
                        fp=open(txt_filename,'w',encoding='utf-8')
                        fp.writelines(txt)
                        fp.close()
                        x.close() 
                        count+=1
                        x.switch_to.window(windows[0])
                    except Exception as e:
                        x.switch_to.window(x.window_handles[0])
                        continue
                #x.find_element_by_class_name('pagination_next').find_element_by_xpath('a').click()
                page+=1
                try:
                    goto_npage(x, section, page)
                except:
                    try:
                        sleep(1)
                        goto_npage(x, section, page)
                    except:
                        break
        logfp.close()
        print('downloaded %d articles.' % count)
        print('crawl from %d to %d' % (From, page) )
    except:
        pass

 
def download_all_section(*arg):
    if len(arg)==1:
        k=arg[0]
        th=[]
        for key in decode.keys():
            th.append(threading.Thread(target=download, args=(key,k)))
        for t in th:
            t.start()
        for t in th:
            t.join()      
    elif len(arg)==2:
        From=arg[0]
        To=arg[1]
        th=[]
        for key in decode.keys():
            th.append(threading.Thread(target=download, args=(key, From, To)))
        for t in th:
            t.start()
        for t in th:
            t.join()

def parallel_download_all_section(*arg):
    if len(arg)==1:
        k=arg[0]
        pro=[]
        for key in decode.keys():
            pro.append(multiprocessing.Process(target=download, args=(key, k)))
            #th.append(threading.Thread(target=download, args=(key,k)))
        for p in pro:
            p.start()
        for p in pro:
            p.join()
    elif len(arg)==2:
        From=arg[0]
        To=arg[1]
        pro=[]
        for key in decode.keys():
            pro.append(multiprocessing.Process(target=download, args=(key, From, To)))
            #th.append(threading.Thread(target=download, args=(key,k)))
        for p in pro:
            p.start()
        for p in pro:
            p.join()

if __name__=='__main__':
    prev_time = datetime.datetime.now() #当前时间
    parallel_download_all_section(1,500)
    cur_time = datetime.datetime.now()  #训练后此时时间
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    print("It costs %02d:%02d:%02d" % (h, m, s))


################################# log #######################################
"""
要闻抓到431页就停了，实际上是可以抓到500页的
国内财经抓到395页就停了，实际上是可以抓到500页的
行业研究抓到257页就停了，实际上是可以抓到500页的
宏观经济研究目前只有236页
"""