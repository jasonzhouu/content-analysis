"""
This is used to control the whole news recommend system operation
"""

from ContentEngine import ContentEngine
import datetime
import pandas as pd
import numpy as np
import jieba.analyse
from sklearn.metrics.pairwise import cosine_similarity
from Naked.toolshed.shell import execute_js


if __name__ == '__main__':
    # initialize jieba
    jieba.analyse.set_stop_words("stopwords.txt")

    my_engine = ContentEngine('localhost', 'root', 'xiaoyu2698355', 'rss')

    # read updated news from database
    now_date = datetime.datetime.now().strftime("%Y-%m-%d")
    # 只对当天录入数据库的文章分析特征向量
    sql = "SELECT id, title, content FROM articles WHERE created_at LIKE " + "'" + now_date + "%';"
    lines = my_engine.execute_sql(sql)

    update_news = pd.DataFrame()  # store update news
    for line in lines:
        # clean news content
        clean_content = my_engine.clean_content(line[2])

        one_news = pd.DataFrame({'newsid': str(line[0]), 'title': line[1], 'content': clean_content}, index=[0])
        update_news = update_news.append(one_news, ignore_index=True)

    # convert news to vector
    news_vector = dict()  # store updated news vectors
    for i in update_news.index:
        news_id = update_news.newsid[i]
        one_title_vector = my_engine.get_news_vector(update_news.title[i])
        one_news_vector = my_engine.get_news_vector(update_news.content[i])
        news_vector[news_id] = one_title_vector + one_news_vector

    # read all user id
    sql = "SELECT id FROM users;"
    user_tuple = my_engine.execute_sql(sql)
    user_list = [str(user_id[0]) for user_id in user_tuple]  # user id list

    # update user interesting model and recommend news
    for user in user_list:
        # read the latest 50 recordings
        sql = "SELECT article_id FROM reading_history_articles" + " WHERE user_id=" + user + " ORDER BY created_at DESC LIMIT 50;"
        rcd_tuple = my_engine.execute_sql(sql)
        rcd_list = [str(rcd[0]) for rcd in rcd_tuple]  # recording id list
        if not rcd_list:
            # if no recordings, continue
            continue;

        # compute eim for each user
        user_eim = np.zeros(len(my_engine.feature_sequence))
        for rcd in rcd_list:
            sql = "SELECT title, content FROM articles WHERE id=" + rcd + ";"
            news = my_engine.execute_sql(sql)
            news_title = news[0][0]
            news_content = my_engine.clean_content(news[0][1])

            content_vector = my_engine.get_news_vector(news_content)
            title_vector = my_engine.get_news_vector(news_title)
            user_eim += content_vector + title_vector

        user_eim = user_eim / len(rcd_list)

        # recommend news
        recommend_result = pd.DataFrame(columns=['newsid', 'similarity'])

        for newsid, one_news_vector in news_vector.items():
            similarity = cosine_similarity(user_eim[np.newaxis, :],
                                            one_news_vector[np.newaxis, :])
            one_result = pd.DataFrame({'newsid': newsid,
                                        'similarity': similarity[0][0]},
                                        index=[0])
            recommend_result = recommend_result.append(one_result, ignore_index=True)

        recommend_result = recommend_result.sort_values(by='similarity', ascending=False)

        # write recommend result to database
        for index, row in recommend_result.iterrows():
            user_id = user
            article_id = row.newsid
            similarity = str(row.similarity)
            created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            sql = "INSERT INTO recommend_articles (article_id, user_id, similarity, created_at)" + " VALUES " + "('" + article_id + "', '" + user_id + "', '" + similarity + "', '" + created_at + "');"
            my_engine.execute_sql(sql, commit=True)

        recommend_result.drop(recommend_result.index, inplace=True)