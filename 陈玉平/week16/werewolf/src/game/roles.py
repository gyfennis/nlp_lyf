from enum import Enum
from dataclasses import dataclass
from typing import Optional


class Role(Enum):
    """狼人杀角色"""
    WOLF = "wolf"           # 狼人
    VILLAGER = "villager"   # 村民
    SEER = "seer"           # 预言家
    WITCH = "witch"         # 女巫
    HUNTER = "hunter"       # 猎人
    IDIOT = "idiot"         # 白痴（暂未实现）


class Team(Enum):
    """阵营"""
    VILLAGE = "village"     # 好人
    WOLF_TEAM = "wolf"      # 狼人


@dataclass
class Player:
    """玩家"""
    id: str
    name: str
    role: Role
    is_alive: bool = True
    is_wolf: bool = False   # 是否为狼人（用于狼人阵营）
    will_vote: Optional[str] = None  # 本轮投票目标
    seer_checked: Optional[str] = None  # 预言家验人结果
    witch_save_target: Optional[str] = None  # 女巫救的人
    witch_poison_target: Optional[str] = None  # 女巫毒的人
    witch_has_save: bool = True    # 女巫是否还有解药
    witch_has_poison: bool = True  # 女巫是否还有毒药
    is_silent: bool = False  # 是否被禁言

    def get_team(self) -> Team:
        if self.role == Role.WOLF:
            return Team.WOLF_TEAM
        return Team.VILLAGE


# 6人局配置
DEFAULT_PLAYER_CONFIG = {
    "wolf": 2,
    "villager": 2,
    "seer": 1,
    "witch": 1,
}
