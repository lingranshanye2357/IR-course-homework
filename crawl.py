import requests
import time
from bs4 import BeautifulSoup
import numpy as np
import re
import os


# 本次的任务和之前那个不一样，这次不需要获取网页链接，东西应该都是现成的
class Crawler(object):
    def __init__(self, crawler_list=[], encoding='utf-8', headers={'user-agent': 'my-app/0.0.1'},
                 wait_time=3, timeout=5, save_path='D:/courses/检索/lofter/', tag_num=1000,
                 base_url='https://www.lofter.com/tag/', mode='total', margin=500):
        self.list = crawler_list    # 爬取序列
        self.encoding = encoding
        self.headers = headers
        self.wait_time = wait_time
        self.timeout = timeout
        self.save_path = save_path
        self.tag_num = tag_num   # 每个tag爬多少个
        self.base_url = base_url
        self.mode = mode        # 爬取模式：全部/月榜/日榜
        self.margin = margin    # 字数限制，一般字数比较多的文章，质量比较高，相关性强

    def crawler(self):
        for tag in self.list:    # 逐个处理tag
            url = self.base_url + tag + '/' + self.mode
            next_page = url
            text_path = self.save_path + '文本/' + tag + '/'
            html_path = self.save_path + '网页/' + tag + '/'
            work_num = 1   # 记录当前tag下爬取的文章数量
            page_num = 1   # 记录当前翻页的数量
            count = 0  # 记录没爬到文章的累积页数

            # 先创建文件夹
            if not os.path.exists(text_path):  # 如果该路径不存在
                os.makedirs(text_path)  # 创造文件夹
            # 创建网页文件夹
            if not os.path.exists(html_path):
                os.makedirs(html_path)

            title_txt = open(text_path + '题目.txt', 'a', encoding=self.encoding)


            while True:
                # 每次循环，分析当前的网页
                try:
                    r = requests.get(url=next_page, headers=self.headers, timeout=self.timeout)
                except:
                    return None
                # 分析该网页的源代码
                html_text = r.text
                soup = BeautifulSoup(html_text, 'html.parser')
                # 提取当前网页中的长文章题目与内容
                titles = soup.find_all('h2', class_="tit")  # 这个是长文章的题目
                contents = soup.find_all('div', class_="txt js-content ptag")  # 这个是长文章的内容
                # 依次提取并保存
                if len(contents) == 0:
                    count += 1
                    print(count)
                if count > 10:  # 连续十页没爬到东西
                    break
                for k in range(len(contents)):   # 一共有len(titles)个长文本
                    title = titles[k].getText()
                    content = contents[k]

                    # 将文章写入
                    with open(text_path + str(work_num) + '.txt', 'w', encoding=self.encoding) as f:
                        f.write(title)
                        f.write('\n')
                        text = '\n'.join([p.text for p in content.find_all('p') if p.text is not None])
                        f.write(text)
                    work_num += 1   # 每保存一篇文章，work数量加一
                    print('tag: {}, work{}: {}'.format(tag, work_num, title))
                    title_txt.write(title)
                    title_txt.write('\n')

                with open(html_path + str(page_num) + '.html', 'w', encoding=self.encoding) as f:
                    f.write(html_text)
                print('page: {}'.format(page_num))
                page_num += 1  # 每处理完一页，页数加一

                if work_num >= self.tag_num:
                    break
                # 当前网页的长文章处理完之后，要获取下一页的url
                next_page = url + '?page=' + str(page_num)
            title_txt.close()


a = Crawler(['ibsm', '云次方', '少爷和我', 'mamamoo', '格林威治', '楼诚', '', '', ''])
a.crawler()
