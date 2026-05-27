import sqlite3
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class GameRecord:
    """游戏记录"""
    id: int
    game_date: str
    day: int
    winners: str
    player_count: int
    log_file: str
    created_at: str


class Storage:
    """存储层"""

    def __init__(self, db_path: str = "data/werewolf.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库"""
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS games (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_date TEXT NOT NULL,
                day INTEGER,
                winners TEXT,
                player_count INTEGER,
                log_file TEXT,
                created_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS player_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_name TEXT NOT NULL,
                role TEXT NOT NULL,
                is_winner INTEGER,
                game_id INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY (game_id) REFERENCES games(id)
            )
        """)

        conn.commit()
        conn.close()

    def save_game(
        self,
        winners: str,
        day: int,
        player_count: int,
        log_file: str,
        player_results: list[dict],
    ):
        """保存游戏结果"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        game_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        created_at = datetime.now().isoformat()

        cursor.execute(
            """
            INSERT INTO games (game_date, day, winners, player_count, log_file, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (game_date, day, winners, player_count, log_file, created_at),
        )
        game_id = cursor.lastrowid

        for result in player_results:
            cursor.execute(
                """
                INSERT INTO player_stats (player_name, role, is_winner, game_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    result["player_name"],
                    result["role"],
                    result["is_winner"],
                    game_id,
                    created_at,
                ),
            )

        conn.commit()
        conn.close()
        return game_id

    def get_recent_games(self, limit: int = 10) -> list[GameRecord]:
        """获取最近的游戏记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, game_date, day, winners, player_count, log_file, created_at
            FROM games
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )

        results = [
            GameRecord(*row)
            for row in cursor.fetchall()
        ]

        conn.close()
        return results

    def get_player_stats(self, player_name: str) -> dict:
        """获取玩家统计数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                role,
                COUNT(*) as games,
                SUM(is_winner) as wins
            FROM player_stats
            WHERE player_name = ?
            GROUP BY role
            """,
            (player_name,),
        )

        stats = {}
        for role, games, wins in cursor.fetchall():
            stats[role] = {
                "games": games,
                "wins": wins,
                "win_rate": wins / games if games > 0 else 0,
            }

        conn.close()
        return stats


storage = Storage()
