"""LLM-driven player draft.

Six manager personas pick one player per round over three rounds. The pick
order is re-randomized every round with a balance constraint: a franchise
never drafts from the same queue position twice, so nobody gets first pick
all tournament (random AND balanced, as required).

Every manager sees only public information: the open player cards, its own
roster and the picks already made by other teams. One pick = one small LLM
call with a structured (enum-constrained) answer. Any LLM failure or invalid
answer falls back to a deterministic auto-pick so a draft always completes.
"""

from __future__ import annotations

import random
from typing import Optional

from .players import compact_card_line


def _balanced_orders(franchise_ids: list[str], rounds: int, rng: random.Random) -> list[list[str]]:
    """One random order per round; no franchise repeats a queue position."""
    orders: list[list[str]] = []
    for _ in range(rounds):
        for _attempt in range(200):
            candidate = list(franchise_ids)
            rng.shuffle(candidate)
            if all(
                all(prev[pos] != fid for prev in orders)
                for pos, fid in enumerate(candidate)
            ):
                orders.append(candidate)
                break
        else:  # pathological rng streak: accept a plain shuffle
            candidate = list(franchise_ids)
            rng.shuffle(candidate)
            orders.append(candidate)
    return orders


def _auto_pick(available: dict[str, dict]) -> str:
    """Deterministic fallback: best available by rating, then potential/attack."""
    def key(card: dict):
        a = card["attributes"]
        return (-card["rating"], -a["potential"], -a["attack"], card["player_id"])
    return sorted(available.values(), key=key)[0]["player_id"]


def _pick_prompt(round_no: int, rounds: int, pick_no: int, roster: list[dict],
                 taken: list[tuple[str, str]], available: dict[str, dict]) -> str:
    roster_line = ", ".join(f"{c['nickname']}({c['player_id']})" for c in roster) or "пусто"
    taken_line = ", ".join(f"{pid}->{team}" for pid, team in taken) or "никто"
    lines = "\n".join(compact_card_line(card) for card in sorted(
        available.values(), key=lambda c: (-c["rating"], c["player_id"])))
    return (
        f"Драфт Regionals: лига 6 команд (round-robin Bo3), затем плей-офф топ-3 (Bo5). "
        f"Раунд {round_no}/{rounds}, твой пик №{pick_no}.\n"
        f"Твой состав: {roster_line}\n"
        f"Уже выбраны другими: {taken_line}\n"
        f"Доступные игроки (id|ник|фракция|архетип|rating|статы 0-100|карьера):\n{lines}\n"
        f"Выбери ОДНОГО игрока (player_id), цель — выиграть турнир. Действуй по своей философии."
    )


_SYSTEM_RULES = (
    " Отвечай только JSON по схеме. Статы: atk атака, def оборона, eco экономика, "
    "tmp темп, con стабильность, pot потенциал."
)


def run_draft(
    template: dict,
    franchise_list: list[dict],
    cards: list[dict],
    rng: random.Random,
    llm: Optional[object] = None,
) -> tuple[dict[str, list[dict]], dict]:
    """Run the full draft. Returns (rosters by franchise_id, draft log)."""
    rounds = int(template["draft_rounds"])
    franchise_ids = [f["franchise_id"] for f in franchise_list]
    personas = {f["franchise_id"]: f for f in franchise_list}
    available: dict[str, dict] = {c["player_id"]: c for c in cards}
    if len(available) < len(franchise_ids) * rounds:
        raise ValueError(
            f"Draft needs {len(franchise_ids) * rounds} players, pool has {len(available)}")

    rosters: dict[str, list[dict]] = {fid: [] for fid in franchise_ids}
    orders = _balanced_orders(franchise_ids, rounds, rng)
    log: dict = {"llm": llm is not None, "rounds": []}

    pick_no = 0
    taken: list[tuple[str, str]] = []  # (player_id, franchise name)
    for round_index in range(rounds):
        order = orders[round_index]
        round_log = {"round": round_index + 1, "order": list(order), "picks": []}
        for fid in order:
            pick_no += 1
            spec = personas[fid]
            player_id, reason, source = None, "", "auto"
            if llm is not None:
                system = spec["persona"] + _SYSTEM_RULES
                prompt = _pick_prompt(round_index + 1, rounds, pick_no,
                                      rosters[fid], taken, available)
                for _attempt in range(2):
                    try:
                        answer = llm.pick(system, prompt, sorted(available))
                        candidate = str(answer.get("player_id", ""))
                        if candidate in available:
                            player_id = candidate
                            reason = str(answer.get("reason", ""))[:200]
                            source = "llm"
                            break
                    except Exception:  # noqa: BLE001 - any LLM failure -> fallback
                        continue
            if player_id is None:
                player_id = _auto_pick(available)
                reason = "автовыбор (LLM недоступен или дал некорректный ответ)"
            card = available.pop(player_id)
            rosters[fid].append(card)
            taken.append((player_id, spec["name"]))
            round_log["picks"].append({
                "pick": pick_no,
                "franchise_id": fid,
                "team_name": spec["name"],
                "manager": spec["manager"],
                "player_id": player_id,
                "nickname": card["nickname"],
                "faction": card["faction"],
                "reason": reason,
                "source": source,
            })
        log["rounds"].append(round_log)

    if llm is not None and hasattr(llm, "usage"):
        log["usage"] = dict(llm.usage)
    return rosters, log
