"""Phase 7 Step 3: loads `config/decision_rules.yaml` into typed, immutable `Rule` objects.

Every rule carries `rule_id`, `version`, `description`, `required_inputs`, `threshold`,
`rationale`, `provenance`, and `status` (`EXPERIMENTAL | PROSPECTIVE_VALIDATION | ACTIVE |
RETIRED`) - the exact fields Step 3 requires. `engine.py` reads specific rules' thresholds
by `rule_id`; it never hardcodes a number that should live here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from nfl_predict.config import get_decision_rules_config


class RuleStatus(str, Enum):
    EXPERIMENTAL = "EXPERIMENTAL"
    PROSPECTIVE_VALIDATION = "PROSPECTIVE_VALIDATION"
    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"


@dataclass(frozen=True)
class Rule:
    rule_id: str
    version: int
    description: str
    required_inputs: tuple[str, ...]
    threshold: object
    rationale: str
    provenance: str
    status: RuleStatus


@dataclass(frozen=True)
class RuleSet:
    rule_set_version: str
    rule_set_status: RuleStatus
    rules: dict[str, Rule]  # keyed by rule_id

    def get(self, rule_id: str) -> Rule:
        try:
            return self.rules[rule_id]
        except KeyError:
            raise KeyError(f"No rule with rule_id={rule_id!r} in rule set {self.rule_set_version!r}") from None


def load_rule_set() -> RuleSet:
    raw = get_decision_rules_config()
    rules = {}
    for r in raw["rules"]:
        rule = Rule(
            rule_id=r["rule_id"], version=r["version"], description=r["description"].strip(),
            required_inputs=tuple(r["required_inputs"]), threshold=r["threshold"],
            rationale=r["rationale"].strip(), provenance=r["provenance"], status=RuleStatus(r["status"]),
        )
        if rule.rule_id in rules:
            raise ValueError(f"Duplicate rule_id in decision_rules.yaml: {rule.rule_id!r}")
        rules[rule.rule_id] = rule
    return RuleSet(rule_set_version=raw["rule_set_version"], rule_set_status=RuleStatus(raw["rule_set_status"]), rules=rules)
