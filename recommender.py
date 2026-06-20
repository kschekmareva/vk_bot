import pandas as pd
import numpy as np
from prepare_models.model_loader import load_models


ease, knn, fusion = load_models()
items = pd.read_csv("items.csv")
popular = pd.read_csv("popular.csv")


def _normalize(scores: dict) -> dict:
    if not scores:
        return {}
    values = np.array(list(scores.values()), dtype=np.float64)
    vmin, vmax = values.min(), values.max()
    normed = (values - vmin) / (vmax - vmin + 1e-10)
    return dict(zip(scores.keys(), normed))


def _get_ease_scores_cold(ease_model, liked_movies, seen):
    """EASE для холодного старта: строим user_vec прямо из лайков,
    не используя user2idx (нового пользователя там нет)."""
    n_items = ease_model.B.shape[0]
    user_vec = np.zeros(n_items, dtype=np.float32)
    seen_idx = []
    for movie in liked_movies:
        idx = ease_model.item2idx.get(int(movie))
        if idx is not None:
            user_vec[idx] = 1.0
            seen_idx.append(idx)

    if not seen_idx:
        return {}

    raw = user_vec @ ease_model.B
    valid = {
        ease_model.idx2item[i]: float(raw[i])
        for i in range(len(raw))
        if ease_model.idx2item[i] not in seen
    }
    return _normalize(valid)


def _get_knn_scores_cold(knn_model, liked_movies, seen):
    seen_idx = [
        knn_model.item2idx[int(movie)]
        for movie in liked_movies
        if int(movie) in knn_model.item2idx
    ]
    if not seen_idx:
        return {}

    raw = np.zeros(knn_model.user_item.shape[1], dtype=np.float32)
    for idx in seen_idx:
        raw[knn_model.sim_idx[idx]] += knn_model.sim[idx]

    valid = {
        knn_model.idx2item[i]: float(raw[i])
        for i in range(len(raw))
        if knn_model.idx2item[i] not in seen
    }
    return _normalize(valid)


def recommend_new_user(fusion_model, liked_movies, n=10):
    """Cold-start рекомендации через Score-Level Fusion (EASE + KNN)."""
    if len(liked_movies) == 0:
        return []

    seen = set(int(m) for m in liked_movies)

    ease_scores = _get_ease_scores_cold(fusion_model.ease_model, liked_movies, seen)
    knn_scores = _get_knn_scores_cold(fusion_model.knn_model, liked_movies, seen)

    all_items = set(ease_scores) | set(knn_scores)
    if not all_items:
        return []

    final_scores = {
        item: fusion_model.w_ease * ease_scores.get(item, 0.0)
              + fusion_model.w_knn * knn_scores.get(item, 0.0)
        for item in all_items
    }

    result = sorted(final_scores, key=final_scores.get, reverse=True)
    return result[:n]


def get_recommendations(liked_movies, disliked_movies, n=10):
    if len(liked_movies) < 2:
        return get_popular(n)
    ids = recommend_new_user(fusion, liked_movies, n)

    movies = items[items["item_id"].isin(ids)]

    movies = movies[~movies["item_id"].isin(disliked_movies)]
    movies = movies[~movies["item_id"].isin(liked_movies)]
    return movies


def get_popular(n=10):
    return popular.head(n)