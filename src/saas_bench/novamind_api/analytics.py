"""Analytics, monitoring, and operations tools."""

from typing import Dict, Optional
from . import _client


def get_social_posts(days: int = 7, limit: int = 50) -> Dict:
    """Get recent social media posts from customers.

    Args:
        days: Number of days to look back (default 7).
        limit: Maximum number of posts to return (default 50).

    Returns:
        Dict with social media post data.
    """
    return _client.call('get_social_posts', {'days': days, 'limit': limit})


def set_targeted_ops_spend(targeted_spend: Dict[str, float]) -> Dict:
    """Set per-group targeted operations spend.

    Args:
        targeted_spend: {group_id: daily_amount} dict.

    Returns:
        Dict with update confirmation.
    """
    return _client.call('set_targeted_ops_spend', {'targeted_spend': targeted_spend})


def set_targeted_dev_spend(targeted_spend: Dict[str, float]) -> Dict:
    """Set per-group targeted development spend.

    Per-group dev spend ACCUMULATES a quality bonus daily. Persists after spending stops.

    Args:
        targeted_spend: {group_id: daily_amount} dict.

    Returns:
        Dict with update confirmation.
    """
    return _client.call('set_targeted_dev_spend', {'targeted_spend': targeted_spend})


def log_rationale(rationale: str) -> Dict:
    """Log your strategic rationale for the day.

    MUST be called exactly once per day, before next-day.

    Args:
        rationale: Your analysis, strategy, and reasoning.

    Returns:
        Dict with logging confirmation.
    """
    return _client.call('log_rationale', {'rationale': rationale})
