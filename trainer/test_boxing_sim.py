import unittest

from trainer.boxing_sim import Action, CombatSupervisor, Fighter


class CombatSupervisorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.supervisor = CombatSupervisor()
        self.fighter = Fighter()

    def test_attack_starts_cooldown_and_costs_stamina(self) -> None:
        decision = self.supervisor.decide(self.fighter, Action.JAB, distance=0.8)
        self.supervisor.apply(self.fighter, decision.executed, delta_seconds=0.02)

        self.assertEqual(decision.executed, Action.JAB)
        self.assertEqual(self.fighter.stamina, 95.0)
        self.assertGreater(self.fighter.cooldown_seconds, 0)

    def test_attack_during_recovery_becomes_guard(self) -> None:
        self.supervisor.apply(self.fighter, Action.JAB, delta_seconds=0.02)

        decision = self.supervisor.decide(self.fighter, Action.CROSS, distance=0.8)

        self.assertEqual(decision.executed, Action.GUARD)
        self.assertEqual(decision.reason, "recovering")

    def test_attack_without_stamina_becomes_guard(self) -> None:
        self.fighter.stamina = 4.0

        decision = self.supervisor.decide(self.fighter, Action.JAB, distance=0.8)

        self.assertEqual(decision.executed, Action.GUARD)
        self.assertEqual(decision.reason, "insufficient_stamina")

    def test_stamina_never_becomes_negative(self) -> None:
        self.fighter.stamina = 1.0
        self.supervisor.apply(self.fighter, Action.ADVANCE, delta_seconds=5.0)

        self.assertEqual(self.fighter.stamina, 0.0)

    def test_repeated_attack_becomes_guard(self) -> None:
        self.fighter.recent_actions = [Action.JAB, Action.JAB, Action.JAB]

        decision = self.supervisor.decide(self.fighter, Action.JAB, distance=0.8)

        self.assertEqual(decision.executed, Action.GUARD)
        self.assertEqual(decision.reason, "repeated_attack")


if __name__ == "__main__":
    unittest.main()