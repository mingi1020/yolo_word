import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


DATABASE_PATH = Path(__file__).with_name("word_explorer.db")
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_]{3,24}$")


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database() -> None:
    with _connection() as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_salt BLOB NOT NULL,
                password_hash BLOB NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS friend_requests (
                id TEXT PRIMARY KEY,
                sender_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                receiver_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'accepted')),
                created_at TEXT NOT NULL,
                CHECK (sender_id <> receiver_id),
                UNIQUE (sender_id, receiver_id)
            );
            CREATE TABLE IF NOT EXISTS photos (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                photo_hash TEXT NOT NULL,
                class_name TEXT NOT NULL,
                korean_word TEXT NOT NULL,
                image BLOB NOT NULL,
                confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
                area_ratio REAL NOT NULL CHECK (area_ratio BETWEEN 0 AND 1),
                created_at TEXT NOT NULL,
                UNIQUE (user_id, photo_hash, class_name)
            );
            CREATE TABLE IF NOT EXISTS battles (
                id TEXT PRIMARY KEY,
                challenger_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                opponent_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                challenger_photo_id TEXT NOT NULL REFERENCES photos(id),
                opponent_photo_id TEXT REFERENCES photos(id),
                scene TEXT NOT NULL DEFAULT 'beach'
                    CHECK (scene IN ('beach', 'forest')),
                status TEXT NOT NULL DEFAULT 'waiting'
                    CHECK (status IN ('waiting', 'finished')),
                winner_id TEXT REFERENCES users(id),
                challenger_hp INTEGER,
                opponent_hp INTEGER,
                fight_log TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                CHECK (challenger_id <> opponent_id)
            );
            CREATE INDEX IF NOT EXISTS friend_requests_receiver_status_idx
                ON friend_requests (receiver_id, status);
            CREATE INDEX IF NOT EXISTS photos_user_created_idx
                ON photos (user_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS battles_opponent_status_idx
                ON battles (opponent_id, status, created_at DESC);
            """
        )
        battle_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(battles)")
        }
        if "scene" not in battle_columns:
            connection.execute(
                "ALTER TABLE battles ADD COLUMN scene TEXT NOT NULL DEFAULT 'beach'"
            )


def create_account(username: str, password: str) -> dict:
    if not USERNAME_PATTERN.fullmatch(username):
        raise ValueError("사용자 이름은 영문, 숫자, 밑줄로 3~24자여야 해요.")
    if len(password) < 8:
        raise ValueError("비밀번호는 8자 이상 입력해 주세요.")

    salt = secrets.token_bytes(16)
    password_hash = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=2**14,
        r=8,
        p=1,
        dklen=32,
        maxmem=64 * 1024 * 1024,
    )
    user_id = str(uuid.uuid4())
    try:
        with _connection() as connection:
            connection.execute(
                """
                INSERT INTO users (id, username, password_salt, password_hash, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    username,
                    salt,
                    password_hash,
                    _now(),
                ),
            )
    except sqlite3.IntegrityError as error:
        raise ValueError("이미 사용 중인 사용자 이름이에요.") from error
    return {"id": user_id, "username": username}


def authenticate(username: str, password: str) -> dict | None:
    with _connection() as connection:
        row = connection.execute(
            """
            SELECT id, username, password_salt, password_hash
            FROM users WHERE username = ? COLLATE NOCASE
            """,
            (username.strip(),),
        ).fetchone()
    if row is None:
        return None

    candidate_hash = hashlib.scrypt(
        password.encode("utf-8"),
        salt=row["password_salt"],
        n=2**14,
        r=8,
        p=1,
        dklen=32,
        maxmem=64 * 1024 * 1024,
    )
    if not hmac.compare_digest(candidate_hash, row["password_hash"]):
        return None
    return {"id": row["id"], "username": row["username"]}


def search_users(username: str, current_user_id: str) -> list[dict]:
    escaped = username.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT id, username FROM users
            WHERE username LIKE ? ESCAPE '\\' COLLATE NOCASE
              AND id <> ?
            ORDER BY username COLLATE NOCASE
            LIMIT 10
            """,
            (f"{escaped}%", current_user_id),
        ).fetchall()
    return [dict(row) for row in rows]


def send_friend_request(sender_id: str, receiver_id: str) -> None:
    if sender_id == receiver_id:
        raise ValueError("자기 자신에게 친구 요청을 보낼 수 없어요.")
    with _connection() as connection:
        existing = connection.execute(
            """
            SELECT status, sender_id FROM friend_requests
            WHERE (sender_id = ? AND receiver_id = ?)
               OR (sender_id = ? AND receiver_id = ?)
            LIMIT 1
            """,
            (sender_id, receiver_id, receiver_id, sender_id),
        ).fetchone()
        if existing:
            raise ValueError("이미 친구이거나 친구 요청이 있어요.")
        connection.execute(
            """
            INSERT INTO friend_requests (id, sender_id, receiver_id, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (str(uuid.uuid4()), sender_id, receiver_id, _now()),
        )


def incoming_friend_requests(user_id: str) -> list[dict]:
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT f.id, f.sender_id, u.username
            FROM friend_requests f
            JOIN users u ON u.id = f.sender_id
            WHERE f.receiver_id = ? AND f.status = 'pending'
            ORDER BY f.created_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def accept_friend_request(request_id: str, receiver_id: str) -> None:
    with _connection() as connection:
        result = connection.execute(
            """
            UPDATE friend_requests SET status = 'accepted'
            WHERE id = ? AND receiver_id = ? AND status = 'pending'
            """,
            (request_id, receiver_id),
        )
        if result.rowcount != 1:
            raise ValueError("친구 요청을 찾을 수 없거나 이미 처리됐어요.")


def friends(user_id: str) -> list[dict]:
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT u.id, u.username
            FROM friend_requests f
            JOIN users u ON u.id = CASE
                WHEN f.sender_id = ? THEN f.receiver_id
                ELSE f.sender_id
            END
            WHERE (f.sender_id = ? OR f.receiver_id = ?)
              AND f.status = 'accepted'
            ORDER BY u.username COLLATE NOCASE
            """,
            (user_id, user_id, user_id),
        ).fetchall()
    return [dict(row) for row in rows]


def save_photo(
    user_id: str,
    photo_hash: str,
    english_word: str,
    korean_word: str,
    image: bytes,
    confidence: float,
    area_ratio: float,
) -> dict:
    with _connection() as connection:
        existing = connection.execute(
            """
            SELECT * FROM photos
            WHERE user_id = ? AND photo_hash = ? AND class_name = ?
            """,
            (user_id, photo_hash, english_word),
        ).fetchone()
        if existing:
            return dict(existing)

        photo_id = str(uuid.uuid4())
        connection.execute(
            """
            INSERT INTO photos (
                id, user_id, photo_hash, class_name, korean_word, image,
                confidence, area_ratio, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                photo_id,
                user_id,
                photo_hash,
                english_word,
                korean_word,
                image,
                confidence,
                area_ratio,
                _now(),
            ),
        )
        return dict(
            connection.execute(
                "SELECT * FROM photos WHERE id = ?",
                (photo_id,),
            ).fetchone()
        )


def list_photos(user_id: str) -> list[dict]:
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT * FROM photos WHERE user_id = ?
            ORDER BY created_at DESC, rowid DESC
            LIMIT 100
            """,
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def create_battle(
    opponent_id: str,
    challenger_id: str,
    photo_id: str,
    scene: str,
) -> str:
    if scene not in {"beach", "forest"}:
        raise ValueError("Choose either the beach or forest arena.")
    battle_id = str(uuid.uuid4())
    with _connection() as connection:
        friendship = connection.execute(
            """
            SELECT 1 FROM friend_requests
            WHERE status = 'accepted'
              AND ((sender_id = ? AND receiver_id = ?)
                OR (sender_id = ? AND receiver_id = ?))
            """,
            (challenger_id, opponent_id, opponent_id, challenger_id),
        ).fetchone()
        if not friendship:
            raise ValueError("친구와만 대전을 시작할 수 있어요.")
        photo = connection.execute(
            "SELECT 1 FROM photos WHERE id = ? AND user_id = ?",
            (photo_id, challenger_id),
        ).fetchone()
        if not photo:
            raise ValueError("선택한 사진을 찾을 수 없어요.")
        connection.execute(
            """
            INSERT INTO battles (
                id, challenger_id, opponent_id, challenger_photo_id, scene, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (battle_id, challenger_id, opponent_id, photo_id, scene, _now()),
        )
    return battle_id


def incoming_battles(user_id: str) -> list[dict]:
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT b.*, u.username AS challenger
            FROM battles b
            JOIN users u ON u.id = b.challenger_id
            WHERE b.opponent_id = ? AND b.status = 'waiting'
            ORDER BY b.created_at DESC
            """,
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def accept_battle(battle_id: str, opponent_id: str, photo_id: str) -> None:
    with _connection() as connection:
        battle = connection.execute(
            """
            SELECT * FROM battles
            WHERE id = ? AND opponent_id = ? AND status = 'waiting'
            """,
            (battle_id, opponent_id),
        ).fetchone()
        if not battle:
            raise ValueError("대전 신청을 찾을 수 없거나 이미 처리됐어요.")

        challenger_photo = connection.execute(
            "SELECT * FROM photos WHERE id = ?",
            (battle["challenger_photo_id"],),
        ).fetchone()
        opponent_photo = connection.execute(
            "SELECT * FROM photos WHERE id = ? AND user_id = ?",
            (photo_id, opponent_id),
        ).fetchone()
        if not challenger_photo or not opponent_photo:
            raise ValueError("대전에 사용할 사진을 찾을 수 없어요.")

        challenger_power = round(
            challenger_photo["confidence"] * 65
            + challenger_photo["area_ratio"] * 35
        )
        opponent_power = round(
            opponent_photo["confidence"] * 65
            + opponent_photo["area_ratio"] * 35
        )
        challenger_hp = opponent_hp = 100
        generator = secrets.SystemRandom()
        fight_log = []
        while challenger_hp > 0 and opponent_hp > 0:
            challenger_damage = (
                7 + generator.randrange(9) + challenger_power // 15
            )
            opponent_hp = max(0, opponent_hp - challenger_damage)
            fight_log.append(
                {
                    "attacker": "challenger",
                    "damage": challenger_damage,
                    "challenger_hp": challenger_hp,
                    "opponent_hp": opponent_hp,
                }
            )
            if opponent_hp == 0:
                break

            opponent_damage = (
                7 + generator.randrange(9) + opponent_power // 15
            )
            challenger_hp = max(0, challenger_hp - opponent_damage)
            fight_log.append(
                {
                    "attacker": "opponent",
                    "damage": opponent_damage,
                    "challenger_hp": challenger_hp,
                    "opponent_hp": opponent_hp,
                }
            )
            if challenger_hp == 0:
                break

        if opponent_hp == 0:
            winner_id = battle["challenger_id"]
        else:
            winner_id = opponent_id

        connection.execute(
            """
            UPDATE battles
            SET opponent_photo_id = ?, status = 'finished', winner_id = ?,
                challenger_hp = ?, opponent_hp = ?, fight_log = ?
            WHERE id = ? AND status = 'waiting'
            """,
            (
                photo_id,
                winner_id,
                challenger_hp,
                opponent_hp,
                json.dumps(fight_log),
                battle_id,
            ),
        )


def recent_battles(user_id: str) -> list[dict]:
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT b.*, challenger.username AS challenger, opponent.username AS opponent
            FROM battles b
            JOIN users challenger ON challenger.id = b.challenger_id
            JOIN users opponent ON opponent.id = b.opponent_id
            WHERE b.status = 'finished'
              AND (b.challenger_id = ? OR b.opponent_id = ?)
            ORDER BY b.created_at DESC
            LIMIT 20
            """,
            (user_id, user_id),
        ).fetchall()
        battles = [dict(row) for row in rows]
        for battle in battles:
            photo_ids = (
                battle["challenger_photo_id"],
                battle["opponent_photo_id"],
            )
            photo_rows = connection.execute(
                f"""
                SELECT id, class_name, korean_word, image FROM photos
                WHERE id IN (?, ?)
                """,
                photo_ids,
            ).fetchall()
            battle["photo_rows"] = [dict(row) for row in photo_rows]
            battle["fight_log"] = json.loads(battle["fight_log"])
    return battles


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
