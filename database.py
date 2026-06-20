import sqlite3

conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
    vk_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    liked TEXT DEFAULT '',
    disliked TEXT DEFAULT ''
)
""")
conn.commit()


def user_exists(vk_id):
    cursor.execute(
        """
        SELECT 1
        FROM users
        WHERE vk_id=?
        """,
        (vk_id,)
    )
    return cursor.fetchone() is not None


def get_user(vk_id):
    cursor.execute(
        """
        SELECT *
        FROM users
        WHERE vk_id=?
        """,
        (vk_id,)
    )
    return cursor.fetchone()


def create_user(vk_id, name):
    cursor.execute(
        """
        INSERT INTO users(
        vk_id,
        name,
        liked,
        disliked)
        VALUES(?,?,?,?)
        """,
        (
            vk_id,
            name,
            "", ""
        )
    )
    conn.commit()


def delete_user(vk_id):
    cursor.execute(
        """
        DELETE FROM users
        WHERE vk_id=?
        """,
        (vk_id,)
    )
    conn.commit()


def add_like(vk_id, movie_id):
    cursor.execute(
        """
        SELECT liked
        FROM users
        WHERE vk_id=?
        """,
        (vk_id,)
    )
    row = cursor.fetchone()
    if not row:
        return
    liked = row[0]
    if liked:
        liked_ids = liked.split(",")
    else:
        liked_ids = []
    liked_ids.append(str(movie_id))

    cursor.execute(
        """
        UPDATE users
        SET liked=?
        WHERE vk_id=?
        """,
        (
            ",".join(liked_ids),
            vk_id
        )
    )
    conn.commit()


def get_likes(vk_id):
    cursor.execute(
        """
        SELECT liked
        FROM users
        WHERE vk_id=?
        """,
        (vk_id,)
    )
    row = cursor.fetchone()
    if not row or not row[0]:
        return []
    return [int(x) for x in row[0].split(",")]


def add_dislike(vk_id, movie_id):
    cursor.execute(
        """
        SELECT disliked
        FROM users
        WHERE vk_id=?
        """,
        (vk_id,)
    )
    row = cursor.fetchone()
    if not row:
        return
    disliked = row[0]
    if disliked:
        ids = disliked.split(",")
    else:
        ids = []

    ids.append(str(movie_id))

    cursor.execute(
        """
        UPDATE users
        SET disliked=?
        WHERE vk_id=?
        """,
        (
            ",".join(ids),
            vk_id
        )
    )
    conn.commit()


def get_dislikes(vk_id):
    cursor.execute(
        """
        SELECT disliked
        FROM users
        WHERE vk_id=?
        """,
        (vk_id,)
    )
    row = cursor.fetchone()
    if not row or not row[0]:
        return []
    return [int(x) for x in row[0].split(",")]
