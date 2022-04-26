import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

from utils import prefilter_items


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, weighting=True):

        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать

        # Популярные покупки юзера
        self.popularity = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.popularity.sort_values('quantity', ascending=False, inplace=True)
        self.popularity = self.popularity[self.popularity['item_id'] != 999999]

        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(
            self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    @staticmethod
    def prepare_matrix(data: pd.DataFrame):

        # your_code

        data = prefilter_items(data, take_n_popular=5000)

        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='quantity',
                                          aggfunc='count',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)

        return user_item_matrix

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model

    def get_recommendations(self, user, model, N=5):
        """Рекомендуем топ-N товаров"""

        res = [self.id_to_itemid[rec[0]] for rec in
               model.recommend(userid=self.userid_to_id[user],
                               user_items=csr_matrix(self.user_item_matrix).tocsr(),
                               N=N,
                               filter_already_liked_items=False,
                               filter_items=[self.itemid_to_id[999999]],
                               recalculate_user=True)]
        return res

    def get_rec_similar_items(self, item_id):
        """Находит похожие товары"""

        smlr_items = self.model.similar_items(self.itemid_to_id[item_id], N=2)
        smlr_items_ = smlr_items[1][0]

        return self.id_to_itemid[smlr_items_]

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        # your_code
        # Практически полностью реализовали на прошлом вебинаре

        popularity_for_user = self.popularity[self.popularity['user_id'] == user].head(N)
        res = popularity_for_user['item_id'].apply(lambda x: self.get_rec_similar_items(x)).tolist()

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)

        return res

    def get_rec_similar_users(self, user_id):
        """Находит похожих юзеров"""

        smlr_users = self.model.similar_users(self.userid_to_id[user_id], N=6)
        smlr_users_ = [i[0] for i in smlr_users]

        return smlr_users_[1:]

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        # your_code

        res = []

        for u in self.get_rec_similar_users(user):
            own_recs = self.get_recommendations(u, self.own_recommender, N=N)
            res.append(own_recs)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)

        return res
