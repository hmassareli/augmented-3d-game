"""Deterministic boxing rules shared conceptually by training and the web game."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Action(StrEnum):
    GUARD = "guard"
    ADVANCE = "advance"
    RETREAT = "retreat"
    CIRCLE_LEFT = "circle_left"
    CIRCLE_RIGHT = "circle_right"
    JAB = "jab"
    CROSS = "cross"
    LEFT_HOOK = "left_hook"
    RIGHT_HOOK = "right_hook"


@dataclass(frozen=True)
class ActionSpec:
    stamina_cost: float = 0.0
    duration_seconds: float = 0.2
    recovery_seconds: float = 0.0
    minimum_range: float = 0.0
    maximum_range: float = 99.0


ACTION_SPECS: dict[Action, ActionSpec] = {
    Action.GUARD: ActionSpec(),
    Action.ADVANCE: ActionSpec(duration_seconds=0.1),
    Action.RETREAT: ActionSpec(duration_seconds=0.1),
    Action.CIRCLE_LEFT: ActionSpec(duration_seconds=0.1),
    Action.CIRCLE_RIGHT: ActionSpec(duration_seconds=0.1),
    Action.JAB: ActionSpec(5.0, 0.18, 0.22, 0.4, 1.25),
    Action.CROSS: ActionSpec(7.0, 0.24, 0.3, 0.4, 1.2),
    Action.LEFT_HOOK: ActionSpec(9.0, 0.3, 0.36, 0.25, 0.85),
    Action.RIGHT_HOOK: ActionSpec(9.0, 0.3, 0.36, 0.25, 0.85),
}


@dataclass
class Fighter:
    stamina: float = 100.0
    cooldown_seconds: float = 0.0
    recent_actions: list[Action] = field(default_factory=list)


@dataclass(frozen=True)
class Decision:
    requested: Action
    executed: Action
    reason: str | None


class CombatSupervisor:
    """Enforces physical/resource rules without selecting the fighter's strategy."""

    stamina_regeneration_per_second = 12.0
    movement_cost_per_second = 3.0
    repetition_window = 6
    repetition_limit = 3

    def decide(self, fighter: Fighter, requested: Action, distance: float) -> Decision:
        spec = ACTION_SPECS[requested]

        if fighter.cooldown_seconds > 0 and spec.stamina_cost > 0:
            return Decision(requested, Action.GUARD, "recovering")

        if spec.stamina_cost > fighter.stamina:
            return Decision(requested, Action.GUARD, "insufficient_stamina")

        if not spec.minimum_range <= distance <= spec.maximum_range:
            return Decision(requested, Action.GUARD, "out_of_range")

        recent_count = fighter.recent_actions[-self.repetition_window :].count(requested)
        if spec.stamina_cost > 0 and recent_count >= self.repetition_limit:
            return Decision(requested, Action.GUARD, "repeated_attack")

        return Decision(requested, requested, None)

    def apply(self, fighter: Fighter, action: Action, delta_seconds: float) -> None:
        spec = ACTION_SPECS[action]
        fighter.cooldown_seconds = max(0.0, fighter.cooldown_seconds - delta_seconds)

        if action in {Action.ADVANCE, Action.RETREAT, Action.CIRCLE_LEFT, Action.CIRCLE_RIGHT}:
            fighter.stamina -= self.movement_cost_per_second * delta_seconds
        elif spec.stamina_cost:
            fighter.stamina -= spec.stamina_cost
            fighter.cooldown_seconds = spec.duration_seconds + spec.recovery_seconds
        else:
            fighter.stamina += self.stamina_regeneration_per_second * delta_seconds

        fighter.stamina = min(100.0, max(0.0, fighter.stamina))
        fighter.recent_actions.append(action)
        del fighter.recent_actions[:-self.repetition_window]