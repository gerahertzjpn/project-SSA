from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


HANDOFF_BACKFILL_DISTANCE = 135.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit replay visual unit continuity.")
    parser.add_argument("replay", type=Path)
    parser.add_argument("--ticks", type=int, default=120)
    parser.add_argument("--substeps", type=int, default=3)
    parser.add_argument("--jump", type=float, default=38.0)
    args = parser.parse_args()

    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    ticks = replay.get("ticks", [])
    if not ticks:
        raise SystemExit("Replay has no ticks")

    samples = []
    max_tick = min(args.ticks, max(0, len(ticks) - 1))
    for tick_index in range(max_tick):
        for substep in range(args.substeps):
            alpha = substep / float(args.substeps)
            units = visual_track_desired(ticks, tick_index, alpha)
            samples.append((tick_index + alpha, units))

    jumps = []
    transitions = Counter()
    speed_spikes = []
    previous_units: dict[str, dict] = {}
    previous_speed: dict[str, float] = {}
    previous_time: float | None = None

    for time, units in samples:
        dt = (time - previous_time) if previous_time is not None else 0.0
        current_by_id = {unit["id"]: unit for unit in units}
        if dt > 0:
            for unit_id, unit in current_by_id.items():
                prev = previous_units.get(unit_id)
                if not prev:
                    continue
                dist = distance(prev, unit)
                speed = dist / dt
                layer_from = prev.get("layer", "")
                layer_to = unit.get("layer", "")
                if layer_from != layer_to:
                    transitions[(layer_from, layer_to)] += 1
                if dist >= args.jump:
                    jumps.append((dist, speed, time, unit_id, layer_from, layer_to, prev, unit))
                old_speed = previous_speed.get(unit_id)
                if old_speed is not None and old_speed > 1e-6 and speed / old_speed >= 2.8 and speed > 45.0:
                    speed_spikes.append((speed / old_speed, speed, old_speed, time, unit_id, layer_from, layer_to, prev, unit))
                previous_speed[unit_id] = speed
        previous_units = current_by_id
        previous_time = time

    print(f"replay={args.replay}")
    print(f"schema={replay.get('meta', {}).get('schema_version')} ticks_scanned={max_tick} substeps={args.substeps}")
    print(f"samples={len(samples)} ids_last={len(previous_units)} jumps>={args.jump:g}={len(jumps)} speed_spikes={len(speed_spikes)}")
    if transitions:
        print("layer_transitions:")
        for (left, right), count in transitions.most_common(12):
            print(f"  {left}->{right}: {count}")

    if jumps:
        print("top_jumps:")
        for dist, speed, time, unit_id, layer_from, layer_to, prev, unit in sorted(jumps, reverse=True)[:30]:
            print(
                f"  t={time:.3f} id={unit_id} {layer_from}->{layer_to} "
                f"dist={dist:.2f} speed={speed:.2f}/tick "
                f"({prev['x']:.1f},{prev['y']:.1f})->({unit['x']:.1f},{unit['y']:.1f}) "
                f"type={unit.get('type')} team={unit.get('team')}"
            )

    if speed_spikes:
        print("top_speed_spikes:")
        for ratio, speed, old_speed, time, unit_id, layer_from, layer_to, prev, unit in sorted(speed_spikes, reverse=True)[:20]:
            print(
                f"  t={time:.3f} id={unit_id} {layer_from}->{layer_to} "
                f"ratio={ratio:.2f} speed={speed:.2f} prev_speed={old_speed:.2f} "
                f"({prev['x']:.1f},{prev['y']:.1f})->({unit['x']:.1f},{unit['y']:.1f})"
            )


def visual_track_desired(ticks: list[dict], tick_index: int, alpha: float) -> list[dict]:
    tick = ticks[tick_index]
    next_tick = ticks[min(len(ticks) - 1, tick_index + 1)]
    previous_tick = ticks[tick_index - 1] if tick_index > 0 else {}
    current = normalize_visual_units(tick.get("visual_units", []))
    previous_by_id = {unit["id"]: unit for unit in normalize_visual_units(previous_tick.get("visual_units", []))}
    next_by_id = {unit["id"]: unit for unit in normalize_visual_units(next_tick.get("visual_units", []))}
    rows = []
    for unit in current:
        next_unit = next_by_id.get(unit["id"])
        item = dict(unit)
        item["x"] = lerp(unit["x"], (next_unit or unit)["x"], alpha)
        item["y"] = lerp(unit["y"], (next_unit or unit)["y"], alpha)
        item["hpRatio"] = lerp(unit["hpRatio"], (next_unit or unit)["hpRatio"], alpha)
        rows.append(item)
    return rows


def normalize_visual_units(units: list[dict]) -> list[dict]:
    rows = []
    for unit in units or []:
        unit_id = unit.get("id")
        try:
            x = float(unit.get("x"))
            y = float(unit.get("y"))
        except (TypeError, ValueError):
            continue
        if not unit_id or not math.isfinite(x) or not math.isfinite(y):
            continue
        rows.append({
            "id": str(unit_id),
            "x": x,
            "y": y,
            "team": unit.get("owner") or unit.get("team") or "NEUTRAL",
            "type": display_unit_code(unit.get("unit_type") or unit.get("type")),
            "hpRatio": float(unit.get("hp_ratio", unit.get("hpRatio", 1.0)) or 0.0),
            "layer": unit.get("layer") or unit.get("state") or "unit",
        })
    return rows


def apply_visual_handoff_backfill(item: dict, previous: dict | None, alpha: float) -> dict:
    if not should_backfill_handoff(previous, item):
        return item
    eased = smoothstep(alpha)
    out = dict(item)
    out["x"] = lerp(previous["x"], item["x"], eased)
    out["y"] = lerp(previous["y"], item["y"], eased)
    out["hpRatio"] = lerp(previous["hpRatio"], item["hpRatio"], eased)
    return out


def should_backfill_handoff(previous: dict | None, item: dict) -> bool:
    if not previous or previous.get("type") != item.get("type"):
        return False
    if team_bucket(previous.get("team")) != team_bucket(item.get("team")):
        return False
    previous_layer = previous.get("layer") or "unit"
    next_layer = item.get("layer") or "unit"
    if previous_layer == next_layer:
        return False
    if distance(previous, item) > HANDOFF_BACKFILL_DISTANCE:
        return False
    return (
        is_battle_layer(previous_layer)
        or is_battle_layer(next_layer)
        or previous_layer == "moving"
        or next_layer == "moving"
    )


def display_unit_code(unit_type: object) -> str:
    value = str(unit_type or "").upper()
    if value in {"VANGUARD", "V"}:
        return "V"
    if value in {"RANGER", "ARCHER", "A", "R"}:
        return "R"
    if value in {"SKIRMISHER", "S"}:
        return "S"
    if value in {"NEUTRAL", "N"}:
        return "N"
    if value in {"CORE", "C"}:
        return "C"
    if value in {"BARON", "LORD", "B", "L"}:
        return "L"
    return value[:1] or "?"


def team_bucket(team: object) -> str:
    value = str(team or "")
    if value in {"NEUTRAL", "BARON"}:
        return "NEUTRAL"
    return "TEAM_B" if value.startswith("B") else "TEAM_A"


def is_battle_layer(layer: object) -> bool:
    return str(layer or "") in {"battle", "fighting", "fallen"}


def distance(left: dict, right: dict) -> float:
    return math.hypot(float(right["x"]) - float(left["x"]), float(right["y"]) - float(left["y"]))


def lerp(left: float, right: float, alpha: float) -> float:
    t = max(0.0, min(1.0, float(alpha)))
    return float(left) + (float(right) - float(left)) * t


def smoothstep(value: float) -> float:
    t = max(0.0, min(1.0, float(value)))
    return t * t * (3.0 - 2.0 * t)


if __name__ == "__main__":
    main()
