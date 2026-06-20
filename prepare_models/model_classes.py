import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

class ItemKNNRecommender:
    def __init__(self, k=50, shrinkage=10.0):
        self.k = k
        self.shrinkage = shrinkage
        self.sim = None
        self.sim_idx = None
        self.user2idx = {}
        self.item2idx = {}
        self.idx2item = {}
        self.user_item = None

    def fit(self, train_df, user_col, item_col):
        self.user_col = user_col
        self.item_col = item_col

        users = train_df[user_col].unique()
        items = train_df[item_col].unique()
        self.user2idx = {int(u): i for i, u in enumerate(users)}
        self.item2idx = {int(it): i for i, it in enumerate(items)}
        self.idx2item = {i: int(it) for it, i in self.item2idx.items()}

        rows = train_df[user_col].map(self.user2idx).values
        cols = train_df[item_col].map(self.item2idx).values
        self.user_item = csr_matrix(
            (np.ones(len(train_df), dtype=np.float32), (rows, cols)),
            shape=(len(users), len(items))
        )
        # item-item схожести
        X = self.user_item.T
        S = cosine_similarity(X, dense_output=False)

        # shrinkage: штрафуем пары с малым числом co-просмотров
        co_counts = (X @ X.T).toarray()
        shrink_factor = co_counts / (co_counts + self.shrinkage)
        S = S.toarray() * shrink_factor
        np.fill_diagonal(S, 0)

        # оставляем топ-K соседей
        print(f'Отбираем топ-{self.k} соседей...')
        self.sim_idx = np.argpartition(S, -self.k, axis=1)[:, -self.k:]
        self.sim = np.take_along_axis(S, self.sim_idx, axis=1)

        print(f'ItemKNN fit: {len(users)} users, {len(items)} items, k={self.k}')
        return self

    def recommend(self, user_id, train_items, n: int = 10):
        user_id = int(user_id)
        if user_id not in self.user2idx:
            return []
        seen_idx = [self.item2idx[int(i)] for i in train_items if int(i) in self.item2idx]
        if not seen_idx:
            return []
        # скор айтема = сумма схожестей с просмотренными
        scores = np.zeros(self.user_item.shape[1], dtype=np.float32)
        for si in seen_idx:
            scores[self.sim_idx[si]] += self.sim[si]

        scores[seen_idx] = -np.inf
        top_idx = np.argpartition(scores, -n)[-n:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [self.idx2item[int(i)] for i in top_idx]

class EASERecommender:
    def __init__(self, l2_lambda=250.0, weight_col=None, normalize_weights=False):
        self.l2_lambda = l2_lambda
        self.weight_col = weight_col
        self.normalize_weights = normalize_weights
        self.B = None # матрица весов
        self.user2idx = {}
        self.item2idx = {}
        self.idx2item = {}
        self.user_item = None

    def fit(self, train_df: pd.DataFrame, user_col: str, item_col: str):
        self.user_col = user_col
        self.item_col = item_col

        users = train_df[user_col].unique()
        items = train_df[item_col].unique()
        self.user2idx = {u: i for i, u in enumerate(users)}
        self.item2idx = {it: i for i, it in enumerate(items)}
        self.idx2item = {i: int(it) for it, i in self.item2idx.items()}
        n_users, n_items = len(users), len(items)

        # веса
        if self.weight_col and self.weight_col in train_df.columns:
            vals = train_df[self.weight_col].astype(np.float32).values
            if self.normalize_weights:
                vmax = vals.max()
                if vmax > 0:
                    vals = vals / vmax
        else:
            vals = np.ones(len(train_df), dtype=np.float32)

        rows = train_df[user_col].map(self.user2idx).values
        cols = train_df[item_col].map(self.item2idx).values

        X = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
        self.user_item = X #сохраняем для инференса

        # X^T*X
        G = (X.T @ X).toarray().astype(np.float64)
        # регуляризация
        G += self.l2_lambda * np.eye(n_items)
        #инвертируем матрицу
        P = np.linalg.inv(G)
        # обнуляем диагональ и нормируем
        B = P / (-np.diag(P))
        np.fill_diagonal(B, 0.0)
        self.B = B
        print(f'EASE fit: {n_users} users, {n_items} items, λ={self.l2_lambda}')
        return self

    def recommend(self, user_id, train_items, n: int = 10):
        if user_id not in self.user2idx:
            return []
        user_idx = self.user2idx[user_id]
        user_vec = self.user_item[user_idx].toarray().ravel()
        scores = user_vec @ self.B
        seen_idx = [self.item2idx[i] for i in train_items if i in self.item2idx]
        scores[seen_idx] = -np.inf
        top_idx = np.argpartition(scores, -n)[-n:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [self.idx2item[int(i)] for i in top_idx]


class ScoreLevelFusion:
    def __init__(self, ease_model, knn_model, w_ease=0.7, w_knn=0.3, n_candidates=100):
        self.ease_model = ease_model
        self.knn_model = knn_model
        self.w_ease = w_ease
        self.w_knn = w_knn
        self.n_candidates = n_candidates

    def _get_ease_scores(self, user_id, seen):
        if user_id not in self.ease_model.user2idx:
            return {}
        user_idx = self.ease_model.user2idx[user_id]
        user_vec = self.ease_model.user_item[user_idx].toarray().ravel()
        raw      = user_vec @ self.ease_model.B
        # Нормализуем в [0,1]
        valid    = {self.ease_model.idx2item[i]: float(raw[i])
                    for i in range(len(raw))
                    if self.ease_model.idx2item[i] not in seen}
        if not valid:
            return {}
        v = np.array(list(valid.values()))
        v = (v - v.min()) / (v.max() - v.min() + 1e-10)
        return dict(zip(valid.keys(), v))

    def _get_knn_scores(self, user_id, train_items, seen):
        seen_idx = [self.knn_model.item2idx[int(i)]
                    for i in train_items if int(i) in self.knn_model.item2idx]
        if not seen_idx:
            return {}
        raw = np.zeros(self.knn_model.user_item.shape[1], dtype=np.float32)
        for si in seen_idx:
            raw[self.knn_model.sim_idx[si]] += self.knn_model.sim[si]
        valid = {self.knn_model.idx2item[i]: float(raw[i])
                 for i in range(len(raw))
                 if self.knn_model.idx2item[i] not in seen}
        if not valid:
            return {}
        v = np.array(list(valid.values()))
        v = (v - v.min()) / (v.max() - v.min() + 1e-10)
        return dict(zip(valid.keys(), v))

    def recommend(self, user_id, train_items, n=10):
        user_id = int(user_id)
        seen = set(int(i) for i in train_items)

        ease_s = self._get_ease_scores(user_id, seen)
        knn_s = self._get_knn_scores(user_id, train_items, seen)

        all_items = set(ease_s) | set(knn_s)
        final = {
            item: self.w_ease * ease_s.get(item, 0.0) +
                  self.w_knn  * knn_s.get(item,  0.0)
            for item in all_items
        }
        return sorted(final, key=final.get, reverse=True)[:n]
