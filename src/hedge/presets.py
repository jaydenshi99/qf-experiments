"""Preset bet types for easy bet creation."""


def generate_call_option_payoff(strike, time_to_expiry, payout_multiplier):
    return f"max(H_{time_to_expiry} - {strike}, 0) * {payout_multiplier} * I - I"


def generate_put_option_payoff(strike, time_to_expiry, payout_multiplier):
    return f"max({strike} - H_{time_to_expiry}, 0) * {payout_multiplier} * I - I"


def generate_binary_call_payoff(strike, time_to_expiry, payout_multiplier):
    return f"max(0, min(1, H_{time_to_expiry} - {strike} + 1)) * {payout_multiplier} * I - I"


def generate_binary_put_payoff(strike, time_to_expiry, payout_multiplier):
    return f"max(0, min(1, {strike} - H_{time_to_expiry} + 1)) * {payout_multiplier} * I - I"


SIMPLE_BET_PRESET = {
    "type": "simple",
    "description": "Condition-based bet with fixed odds (win/lose)",
    "params": {
        "condition": {"label": "Condition", "help": "Payout condition (e.g., 'H_2 == 1', 'H_5 >= 3', 'H_5 < 10')", "default": "H_1 == 1"},
        "odds": {"label": "Odds", "help": "x : 1 odds (e.g., 2.0 = 2x payout)", "min": 0.01, "default": 1.0, "step": 0.05}
    }
}

PRESET_TYPES = {
    "Simple Bet": SIMPLE_BET_PRESET,
    "Call Option": {
        "function": generate_call_option_payoff,
        "description": "Pays out per head above strike price",
        "params": {
            "strike": {"label": "Strike Price", "help": "Number of heads threshold", "min": 0, "default": 3},
            "time_to_expiry": {"label": "Time to Expiry", "help": "Number of coin flips", "min": 1, "default": 5},
            "payout_multiplier": {"label": "Payout per Head", "help": "Payout per head above strike (e.g., 1 = $1)", "min": 0.01, "default": 1.0, "step": 0.1}
        }
    },
    "Put Option": {
        "function": generate_put_option_payoff,
        "description": "Pays out per head below strike price",
        "params": {
            "strike": {"label": "Strike Price", "help": "Number of heads threshold", "min": 0, "default": 3},
            "time_to_expiry": {"label": "Time to Expiry", "help": "Number of coin flips", "min": 1, "default": 5},
            "payout_multiplier": {"label": "Payout per Head", "help": "Payout per head below strike (e.g., 1 = $1)", "min": 0.01, "default": 1.0, "step": 0.1}
        }
    },
    "Binary Call": {
        "function": generate_binary_call_payoff,
        "description": "All-or-nothing payout if heads >= strike",
        "params": {
            "strike": {"label": "Strike Price", "help": "Number of heads threshold", "min": 0, "default": 3},
            "time_to_expiry": {"label": "Time to Expiry", "help": "Number of coin flips", "min": 1, "default": 5},
            "payout_multiplier": {"label": "Payout Multiplier", "help": "Total payout if condition met (e.g., 2.0 = 2x investment)", "min": 0.01, "default": 2.0, "step": 0.1}
        }
    },
    "Binary Put": {
        "function": generate_binary_put_payoff,
        "description": "All-or-nothing payout if heads <= strike",
        "params": {
            "strike": {"label": "Strike Price", "help": "Number of heads threshold", "min": 0, "default": 3},
            "time_to_expiry": {"label": "Time to Expiry", "help": "Number of coin flips", "min": 1, "default": 5},
            "payout_multiplier": {"label": "Payout Multiplier", "help": "Total payout if condition met (e.g., 2.0 = 2x investment)", "min": 0.01, "default": 2.0, "step": 0.1}
        }
    }
}

