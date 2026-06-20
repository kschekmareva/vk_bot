import pandas as pd
import vk_api
from vk_api.longpoll import VkLongPoll, VkEventType
from vk_api.keyboard import VkKeyboard, VkKeyboardColor
from vk_api.utils import get_random_id

from database import (create_user, user_exists, get_user, delete_user, add_like, get_likes, add_dislike, get_dislikes)
from recommender import get_recommendations
from config import TOKEN

popular_df = pd.read_csv("popular.csv")
items_df = pd.read_csv("items.csv")

popular_df = popular_df.merge(items_df[["item_id", "description"]],
                              on="item_id", how="left")

vk_session = vk_api.VkApi(token=TOKEN)
vk = vk_session.get_api()
longpoll = VkLongPoll(vk_session)
print("Бот запущен")

registration_state = {}
movie_test_state = {}
recommend_pages = {}
support_state = {}
user_pages = {}


def main_keyboard():
    keyboard = VkKeyboard(one_time=False)
    keyboard.add_button("Подобрать фильмы", color=VkKeyboardColor.POSITIVE)
    keyboard.add_button("Популярные фильмы", color=VkKeyboardColor.PRIMARY)

    keyboard.add_line()
    keyboard.add_button("О боте", color=VkKeyboardColor.SECONDARY)
    keyboard.add_button("Поддержка", color=VkKeyboardColor.SECONDARY)

    keyboard.add_line()
    keyboard.add_button("Удалить аккаунт", color=VkKeyboardColor.NEGATIVE)
    return keyboard.get_keyboard()


def genre_keyboard():
    keyboard = VkKeyboard(one_time=False)

    keyboard.add_button("Драмы", color=VkKeyboardColor.PRIMARY)
    keyboard.add_button("Комедии", color=VkKeyboardColor.PRIMARY)
    keyboard.add_line()
    keyboard.add_button("Боевики", color=VkKeyboardColor.PRIMARY)
    keyboard.add_button("Фантастика", color=VkKeyboardColor.PRIMARY)
    keyboard.add_line()
    keyboard.add_button("Триллеры", color=VkKeyboardColor.PRIMARY)
    keyboard.add_button("Мелодрамы", color=VkKeyboardColor.PRIMARY)
    keyboard.add_line()
    keyboard.add_button("Ужасы", color=VkKeyboardColor.PRIMARY)
    keyboard.add_button("Фэнтези", color=VkKeyboardColor.PRIMARY)
    keyboard.add_line()
    keyboard.add_button("Приключения", color=VkKeyboardColor.PRIMARY)
    keyboard.add_button("Детективы", color=VkKeyboardColor.PRIMARY)

    keyboard.add_line()
    keyboard.add_button("🏠 В меню", color=VkKeyboardColor.SECONDARY)
    return keyboard.get_keyboard()


def rating_keyboard(can_recommend=False):
    keyboard = VkKeyboard(one_time=False)
    keyboard.add_button("👍 Нравится", color=VkKeyboardColor.POSITIVE)
    keyboard.add_button("👎 Не нравится", color=VkKeyboardColor.NEGATIVE)
    keyboard.add_button("😐 Не смотрел(а)", color=VkKeyboardColor.SECONDARY)
    keyboard.add_line()
    if can_recommend:
        keyboard.add_button("🎯 Рекомендации", color=VkKeyboardColor.PRIMARY)
        keyboard.add_line()

    keyboard.add_button("🏠 В меню", color=VkKeyboardColor.SECONDARY)
    return keyboard.get_keyboard()


def movies_keyboard():
    keyboard = VkKeyboard(one_time=False)
    keyboard.add_button("Еще популярные", color=VkKeyboardColor.PRIMARY)
    keyboard.add_button("Назад", color=VkKeyboardColor.NEGATIVE)
    return keyboard.get_keyboard()


def recommendations_keyboard():
    keyboard = VkKeyboard(one_time=False)
    keyboard.add_button("Посмотреть еще", color=VkKeyboardColor.PRIMARY)
    keyboard.add_button("Назад", color=VkKeyboardColor.NEGATIVE)
    return keyboard.get_keyboard()


def send_message(user_id, message, keyboard=None):
    vk.messages.send(
        user_id=user_id,
        message=message,
        random_id=get_random_id(),
        keyboard=keyboard
    )


def show_popular_movies(user_id, page=0):
    start = page * 5
    end = start + 5
    movies = popular_df.iloc[start:end]

    if movies.empty:
        send_message(user_id, "Фильмы закончились 😢", main_keyboard())
        return

    text = "🎬 Популярные фильмы:\n\n"
    for i, (_, row) in enumerate(movies.iterrows(), start=start+1):
        description = row.get( "description", "Описание отсутствует")
        if pd.isna(description):
            description = "Описание отсутствует"
        description = str(description)
        if len(description) > 200: # бот не может выводить полные большие описания из-за ограничений в длине сообщения
            description = description[:200] + "..."
        text += (f"{i}. 🎬 {row['title']}\n"
                 f"Жанры: {row['genres']}\n"
                 f"📝 {description}\n\n")
    send_message(user_id, text, movies_keyboard())


def start_movie_test(user_id):
    movie_test_state[user_id] = {"state": "waiting_genre"}
    send_message(user_id, "🎬 Какой жанр фильмов вам нравится?\n\n"
                          "Выберите жанр:", genre_keyboard())


def start_genre_movies(user_id, genre):
    genre = genre.lower().strip()
    movies = items_df[items_df["genres"].fillna("").str.lower().str.contains(genre, regex=False)]
    print(f"Жанр: {genre}. Найдено фильмов: {len(movies)}")

    if movies.empty:
        send_message(user_id, "Фильмы не найдены 😢\n"
                              "Попробуйте другой жанр.", genre_keyboard())
        return

    movie_ids = (movies["item_id"].tolist())
    movie_test_state[user_id] = {"state": "rating", "genre": genre, "movies": movie_ids, "index": 0}

    send_next_movie(user_id)


def send_next_movie(user_id):
    state = movie_test_state[user_id]
    index = state["index"]

    if index >= len(state["movies"]):
        state["index"] = 0
        index = 0

    movie_id = state["movies"][index]
    movie = items_df[items_df["item_id"] == movie_id].iloc[0]
    likes = get_likes(user_id)
    send_message(user_id, "🎬 Оцените фильм (чем больше лайков, тем точнее рекомендации):\n\n"
                          f"{movie['title']}\n\n"
                          f"Жанры: {movie['genres']}",
                 rating_keyboard(can_recommend=len(likes) >= 2))


def show_recommendations(user_id):
    likes = get_likes(user_id)
    dislikes = get_dislikes(user_id)
    if len(likes) < 2:
        send_message(user_id,"Нужно минимум 2 понравившихся фильма "
                             "для персональных рекомендаций.",main_keyboard())
        return

    page = recommend_pages.get(user_id, 0)
    recs = get_recommendations(likes, dislikes, n=100)

    start = page * 5
    end = start + 5
    movies = recs.iloc[start:end]

    if movies.empty:
        send_message(user_id, "Больше рекомендаций нет 😢", main_keyboard())
        return

    text = ("🎯 Ваши персональные рекомендации:\n\n")
    for i, (_, row) in enumerate(movies.iterrows(),start=start+1):
        description = row.get("description", "Описание отсутствует")
        if pd.isna(description):
            description = "Описание отсутствует"
        description = str(description)
        if len(description) > 200:
            description = description[:200] + "..."
        text += (f"{i}. 🎬 {row['title']}\n"
                 f"Жанры: {row['genres']}\n"
                 f"📝 {description}\n\n")

    send_message(user_id, text, recommendations_keyboard())


while True:
    try:
        for event in longpoll.listen():
            if event.type == VkEventType.MESSAGE_NEW and event.to_me:
                user_id = event.user_id
                text = event.text.lower().strip()
                print(user_id, text)

                if not user_exists(user_id):
                    if user_id not in registration_state:
                        registration_state[user_id] = True
                        send_message(user_id, "👋 Привет!\n"
                                              "Для регистрации напишите ваше имя.")
                        continue
                    else:
                        create_user(user_id, text)
                        registration_state.pop(user_id)

                        send_message(user_id, "Регистрация завершена 🎉\n"
                                              "Выберите необходимую опцию в меню.",
                                     main_keyboard())
                        continue

                if user_id in movie_test_state and movie_test_state[user_id].get("state") == "waiting_genre":
                    if text == "🏠 в меню":
                        movie_test_state.pop(user_id, None)
                        send_message(user_id, "Главное меню", main_keyboard())
                        continue
                    start_genre_movies(user_id, text)
                    continue

                if text in ["привет", "начать", "start"]:
                    user = get_user(user_id)
                    send_message(user_id, f"Привет, {user[1].capitalize()} 👋", main_keyboard())

                elif text == "подобрать фильмы":
                    start_movie_test(user_id)

                elif text == "👍 нравится":
                    if user_id not in movie_test_state:
                        continue
                    state = movie_test_state[user_id]
                    movie_id = state["movies"][state["index"]]
                    add_like(user_id, movie_id)
                    state["index"] += 1
                    send_next_movie(user_id)

                elif text == "👎 не нравится":
                    if user_id not in movie_test_state:
                        continue
                    state = movie_test_state[user_id]
                    movie_id = state["movies"][state["index"]]
                    add_dislike(user_id, movie_id)
                    state["index"] += 1
                    send_next_movie(user_id)

                elif text == "😐 не смотрел(а)":
                    if user_id not in movie_test_state:
                        continue
                    movie_test_state[user_id]["index"] += 1
                    send_next_movie(user_id)

                elif text == "🎯 рекомендации":
                    likes = get_likes(user_id)
                    dislikes = get_dislikes(user_id)

                    if len(likes) < 2: # при наличии 2 и более "лайков" - подключаем модель
                        send_message(user_id, "Сначала оцените хотя бы 2 фильма 👍")
                        continue

                    recommend_pages[user_id] = 0
                    show_recommendations(user_id)

                elif text == "популярные фильмы":
                    show_popular_movies(user_id)

                elif text == "посмотреть еще":
                    recommend_pages[user_id] = (recommend_pages.get(user_id, 0) + 1)
                    show_recommendations(user_id)

                elif text == "еще популярные":
                    user_pages[user_id] = (user_pages.get(user_id, 0) + 1)
                    show_popular_movies(user_id, user_pages[user_id])

                elif text == "🏠 в меню":
                    movie_test_state.pop(user_id, None)
                    send_message(user_id, "Главное меню", main_keyboard())

                elif text == "о боте":
                    send_message(user_id,
                                 "Я бот-рекомендатель фильмов 🎬\n"
                                 "Сначала я узнаю ваши интересы,\n"
                                 "а затем подберу персональные рекомендации\n"
                                 "Также можно посмотреть популярные фильмы и сериалы",
                                 main_keyboard())

                elif text == "поддержка":
                    support_state[user_id] = True
                    send_message(user_id,
                                 "💬 Напишите ваш вопрос или пожелание.\n\n"
                                 "Мы обязательно рассмотрим сообщение и свяжемся с вами.")

                elif text == "удалить аккаунт":
                    delete_user(user_id)
                    movie_test_state.pop(user_id, None)
                    recommend_pages.pop(user_id, None)
                    user_pages.pop(user_id, None)
                    support_state.pop(user_id, None)
                    send_message(user_id,
                                 "Аккаунт удалён.\n"
                                 "Напишите любое сообщение для новой регистрации.")

                elif text == "назад":
                    movie_test_state.pop(user_id, None)
                    recommend_pages.pop(user_id, None)
                    user_pages.pop(user_id, None)
                    support_state.pop(user_id, None)
                    send_message(user_id, "Главное меню", main_keyboard())

                else:
                    send_message(user_id, "Не понял команду 😢\n"
                                          "Выберите команду из предложенных или свяжитесь с поддержкой.",
                                 main_keyboard())
    except Exception as e:
        print("Ошибка:", e)