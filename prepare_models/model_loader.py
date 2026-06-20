import pickle
import sys
from prepare_models.model_classes import EASERecommender, ItemKNNRecommender, ScoreLevelFusion

# создаём подмену для старого пути pickle
sys.modules["__main__"].EASERecommender = EASERecommender
sys.modules["__main__"].ItemKNNRecommender = ItemKNNRecommender
sys.modules["__main__"].ScoreLevelFusion = ScoreLevelFusion


def load_models():
    with open("models/ease.pkl", "rb") as f:
        ease = pickle.load(f)

    with open("models/knn.pkl", "rb") as f:
        knn = pickle.load(f)

    with open("models/fusion.pkl", "rb") as f:
        fusion = pickle.load(f)
    return ease, knn, fusion