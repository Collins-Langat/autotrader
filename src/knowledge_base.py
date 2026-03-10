"""
Knowledge Base Loader — reads and validates all JSON/YAML files in knowledge_base/.
Provides the trading rules and strategy definitions to the decision engine.
"""

import json
import os
from pathlib import Path
from typing import Any
import yaml


KB_PATH = Path(__file__).parent.parent / 'knowledge_base'


def _load_json(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_yaml(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class KnowledgeBase:
    """
    Loads and provides access to all trading knowledge:
    - Strategies (JSON)
    - Rules: hard risk, entry, exit (YAML)
    - Market regime playbooks (YAML)
    - Source summaries (Markdown — loaded as text)
    """

    def __init__(self):
        self.strategies: dict[str, dict] = {}
        self.hard_risk_rules: list[dict] = []
        self.entry_rules: list[dict] = []
        self.exit_rules: list[dict] = []
        self.market_regimes: dict[str, dict] = {}
        self.sources: dict[str, str] = {}
        self._load_all()

    def _load_all(self):
        self._load_strategies()
        self._load_rules()
        self._load_market_regimes()
        self._load_sources()

    def _load_strategies(self):
        strategies_path = KB_PATH / 'strategies'
        for file in strategies_path.glob('*.json'):
            strategy = _load_json(file)
            self.strategies[strategy['name']] = strategy

    def _load_rules(self):
        rules_path = KB_PATH / 'rules'
        hard = _load_yaml(rules_path / 'hard_risk_rules.yaml')
        self.hard_risk_rules = hard.get('rules', [])
        entry = _load_yaml(rules_path / 'entry_rules.yaml')
        self.entry_rules = entry.get('rules', [])
        exit_ = _load_yaml(rules_path / 'exit_rules.yaml')
        self.exit_rules = exit_.get('rules', [])

    def _load_market_regimes(self):
        regimes_path = KB_PATH / 'market_regimes'
        for file in regimes_path.glob('*.yaml'):
            regime = _load_yaml(file)
            self.market_regimes[file.stem] = regime

    def _load_sources(self):
        sources_path = KB_PATH / 'sources'
        for file in sources_path.glob('*.md'):
            with open(file, 'r', encoding='utf-8') as f:
                self.sources[file.stem] = f.read()

    def get_strategy(self, name: str) -> dict:
        """Get strategy definition by name."""
        return self.strategies.get(name)

    def get_strategy_entry_conditions(self, name: str) -> dict:
        strategy = self.get_strategy(name)
        if strategy:
            return strategy.get('entry_conditions', {})
        return {}

    def get_hard_risk_rule(self, rule_id: str) -> dict:
        for rule in self.hard_risk_rules:
            if rule.get('id') == rule_id:
                return rule
        return {}

    def get_market_regime(self, regime_name: str) -> dict:
        return self.market_regimes.get(regime_name, {})

    def determine_regime(self, vix: float, avg_iv_rank: float) -> str:
        """Determine current market regime based on VIX and IV rank."""
        if vix > 25 or avg_iv_rank > 60:
            return 'high_iv_environment'
        elif vix < 15 or avg_iv_rank < 30:
            return 'low_iv_environment'
        else:
            return 'normal_environment'

    def get_preferred_strategies_for_regime(self, regime_name: str) -> list:
        regime = self.get_market_regime(regime_name)
        return regime.get('preferred_strategies', [])

    def list_strategies(self) -> list:
        return list(self.strategies.keys())

    def summary(self) -> dict:
        return {
            'strategies_loaded': len(self.strategies),
            'strategy_names': list(self.strategies.keys()),
            'hard_risk_rules': len(self.hard_risk_rules),
            'entry_rules': len(self.entry_rules),
            'exit_rules': len(self.exit_rules),
            'market_regimes': len(self.market_regimes),
            'sources': list(self.sources.keys()),
        }
