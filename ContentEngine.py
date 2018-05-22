import numpy as np
import jieba
import jieba.analyse
import pymysql
import json
import re


class ContentEngine:
    """
        compute user existing interest model (EIM)
        compute similarity between EIM and news
        sort the news
    """

    def __init__(self, host, user, password, database):
        self.host = host
        self.user = user
        self.password = password
        self.database = database

        with open('feature_sequence_long.json', 'r', encoding='utf-8') as fid:
            self.feature_sequence = json.load(fid)

    def execute_sql(self, sql, commit=False):
        """
        execute mysql order and return result
        :param sql: input mysql order
        :param commit: execute commit
        :return: execute result
        """
        conn = pymysql.connect(self.host, self.user, self.password, self.database, charset='utf8')
        cur = conn.cursor()

        cur.execute(sql)
        execute_result = cur.fetchall()
        if commit:
            conn.commit()

        cur.close()
        conn.close()
        return execute_result

    def clean_content(self, raw_content):
        """
        clean news content, remove html tags
        :param raw_content:
        :return:
        """
        cleaner = re.compile('<.*?>')
        clean_content = re.sub(cleaner, '', raw_content)
        return clean_content

    def get_news_vector(self, news):
        """
        transform a news to a vector
        :param news: news content
        :return: news vector
        """
        feature_words = jieba.analyse.extract_tags(news, topK=None,
                                                   allowPOS=['a', 'n', 'v'],
                                                   withWeight=True)
        news_vector = np.zeros(len(self.feature_sequence))
        for fword in feature_words:
            if fword[0] in self.feature_sequence:
                col = self.feature_sequence.index(fword[0])
                news_vector[col] += fword[1]
        return news_vector
