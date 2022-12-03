import requests
import time
from bs4 import BeautifulSoup
from url_normalize import url_normalize
from urllib.parse import urljoin
import numpy as np
import re

class Crawler(object):
    def __init__(self, start_tasks=[], encoding=None, headers={'user-agent': 'my-app/0.0.1'},
                 wait_time=0.05, timeout=5, save_path='D:/courses/检索/作业/'):
        # 设置全局变量
        self.visited = []
        self.queue = start_tasks  # 队列,list

        # self.visited = np.load('./visited_list.npy', allow_pickle=True).tolist()  # set
        # self.queue = np.load('./list.npy', allow_pickle=True).tolist()
        # 基本变量
        self.encoding = encoding
        self.headers = headers
        self.wait_time = wait_time
        self.timeout = timeout
        self.save_path = save_path

    # 网页规范化，爬取不同网页时应该有不同的设置
    def check(self, url, u):
        if 'pdf' in u:
            return None
        if 'download' in u:
            return None
        if 'doc' in u:
            return None
        if 'xlsx' in u:
            return None
        if 'xls' in u:
            return None
        if 'jpg' in u:
            return None
        if 'rar' in u:
            return None
        if 'zip' in u:
            return None
        if 'upload' in u:
            return None
        if '附件' in u:
            return None
        if 'ico' in u:
            return None
        if 'css' in u:
            return None
        if 'uploads' in u:
            return None
        if 'http' not in u:   # 如果不是绝对路径
            u = urljoin(url, u)
            u = url_normalize(u)
            if 'http://ctd.ruc.edu.cn' in u:
                if 'http://ctd.ruc.edu.cn/web/' not in u:
                    u = u.replace('http://ctd.ruc.edu.cn/', 'http://ctd.ruc.edu.cn/web/')
                    return u
            if 'http://fangxue.ruc.edu.cn' in u:
                if 'http://fangxue.ruc.edu.cn/web/' not in u:
                    u = u.replace('http://fangxue.ruc.edu.cn/', 'http://fangxue.ruc.edu.cn/web/')
                    return u
        if 'ruc.edu.cn' in u:  # 检查是否为人大域名下的
            return u
        else:
            return None    #

    def get_html(self, url):
        try:
            r = requests.get(url, headers=self.headers, timeout=3)
            if r.status_code == 404:
                return None
            if self.encoding is not None:
                r.encoding = self.encoding
            return r.text
        except:
            return None

    # 得到一个网页中的所有链接
    def get_url(self, html_text):
        temp_set = set()  # 用集合的方式，防止放入很多重复的
        soup = BeautifulSoup(html_text, 'html.parser')   # 创建一个类
        # soup = BeautifulSoup(html_text, features="lxml-xml")
        for tag in soup.find_all('a'):   # 找到所有的a
            url = tag.attrs.get('href', None)
            if url is not None and url != '':
                temp_set.add(url)
        # for tag in soup.find_all('link'):
        #     url = tag.attrs.get('href', None)
        #     if url is not None and url != '':
        #         temp_set.add(url)
        for tag in soup.find_all('script'):
            a = str(tag)
            if 'href' in a:
                pattern = re.compile("\'(.*?)\'", re.S)  # 表达式为: (.*?)
                url = pattern.findall(a)
            else:
                url = None
            if url is not None:
                try:
                    temp_set.add(url[0])
                except:
                    continue
        return temp_set

    # 得到一个html的正文
    def get_text(self, html_text):
        soup = BeautifulSoup(html_text, 'html.parser')
        # 提取并处理正文与标题
        if soup.title is not None:  # 如果不是空的，就存title
            title = soup.title.string
        else:
            title = None  # 否则title=None

        text = '\n'.join([p.text for p in soup.find_all('p') if p.text is not None])  # 得到正文
        return title, text

    # 爬取
    def crawl(self):
        count = 0   # count用于记录当前的总爬取次数，并为保存的内容命名
        while len(self.queue) > 0:
            url = self.queue.pop(0)  # 弹出第一个
            if 'linkedin' in url:
                continue
            if url in self.visited:
                continue
            html_text = self.get_html(url)  # 第一个函数：得到网页内容
            if html_text == '' or html_text is None:
                continue
            temp_set = self.get_url(html_text)   # 第二个函数：得到网页链接
            for u in temp_set:
                u = self.check(url, u)  # 第三个函数：判断是否已经爬取过
                if (u not in self.queue) and (u not in self.visited) and (u is not None):
                    self.queue.append(u)  # 队列任务加一

            count += 1   # 保存当前网页和正文，count用于计数
            path1 = self.save_path + '网页' + '/' + str(count) + '.html'  # 路径1，存网页
            # path2 = self.save_path + '文本' + '/' + str(count) + '.txt'  # 路径2，存正文
            # path3 = self.save_path + '源码' + '/' + str(count) + '.txt'

            # title, text = self.get_text(html_text)  # 第三个函数：提取标题和正文
            with open(path1, 'w', encoding='utf-8') as f:  # 保存网页
                f.write(html_text)

            # with open(path2, 'w', encoding='utf-8') as f:  # 保存正文
            #     if title is not None:
            #         f.write(title)
            #     f.write('\n')
            #     if text is not None:
            #         f.write(text)
            #
            # with open(path3, 'w', encoding='utf-8') as f:  # 保存源码
            #     f.write(html_text)

            # 先保存正文、网页再加入url，保证一一对应
            print(count, url)
            self.visited.append(url)  # 加入到已访问当中
            np.save(self.save_path + 'list.npy', list(self.queue))  # 保存当前队列，为了下次能从结束的位置开始
            np.save(self.save_path + 'visited_list.npy', list(self.visited))  # 保存已经访问过的网址
            with open(self.save_path + 'html.txt', 'a', newline='') as f:
                f.write(url)
                f.write('\n')

            time.sleep(self.wait_time)  # 每爬取一个网页，需要停顿几秒



a = Crawler(['http://www.ruc.edu.cn/'], encoding='utf-8')