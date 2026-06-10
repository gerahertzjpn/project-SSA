"""Regionals: league + playoff tournament mode with LLM managers and commentator.

Built on top of the league layer (`src/league`): six fixed franchises whose
LLM manager personas draft 3 players each from the 18-net player pool, play a
round-robin Bo3 league, then a Bo5 step-ladder playoff (2nd vs 3rd, winner vs
1st). Adds chess-style Elo ratings with impact modifiers, prize money,
persistent franchise/player registries with head-to-head history, and a
memory-keeping LLM commentator.
"""

from .templates import REGIONALS_TEMPLATES, get_template, list_templates
from .players import (
    ensure_player_store,
    player_cards,
    save_player_store,
    PlayersError,
)
from .franchises import ensure_franchise_store, franchise_specs, save_franchise_store
from .llm import AnthropicLLM, LLMUnavailable, default_llm, llm_unavailable_reason
from .regionals import (
    RegionalsError,
    add_commentary,
    advance_day,
    create_regionals,
    encounter_detail,
    list_regionals,
    load_regionals,
    regionals_public_view,
)

__all__ = [
    "REGIONALS_TEMPLATES",
    "get_template",
    "list_templates",
    "ensure_player_store",
    "player_cards",
    "save_player_store",
    "PlayersError",
    "ensure_franchise_store",
    "franchise_specs",
    "save_franchise_store",
    "AnthropicLLM",
    "LLMUnavailable",
    "default_llm",
    "llm_unavailable_reason",
    "RegionalsError",
    "add_commentary",
    "advance_day",
    "create_regionals",
    "encounter_detail",
    "list_regionals",
    "load_regionals",
    "regionals_public_view",
]
