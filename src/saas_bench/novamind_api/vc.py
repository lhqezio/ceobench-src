"""VC fundraising and equity management tools."""

from typing import Dict, List
from . import _client


def list_potential_vcs() -> Dict:
    """List all potential VC investors and their status.

    Returns:
        Dict with VC investor details.
    """
    return _client.call('list_potential_vcs')


def send_vc_deal(deals: List[Dict]) -> Dict:
    """Send equity offers to VC investors.

    Each deal: {shareholder_id, share_pct, optional term sheet proposals}.

    Args:
        deals: List of deal dicts.

    Returns:
        Dict with per-deal results.
    """
    return _client.call('send_vc_deal', {'deals': deals})


def reject_vc_deal(deals: List[Dict]) -> Dict:
    """Reject VC deal proposals. No penalty.

    Args:
        deals: List of deal rejection dicts.

    Returns:
        Dict with per-deal results.
    """
    return _client.call('reject_vc_deal', {'deals': deals})


def get_cap_table_info() -> Dict:
    """Get current cap table (shareholders, ownership percentages).

    Returns:
        Dict with cap table details.
    """
    return _client.call('get_cap_table_info')


def settle_investments() -> Dict:
    """Settle pending VC investments (dilute founder shares).

    Returns:
        Dict with settlement results.
    """
    return _client.call('settle_investments')


def declare_dividend(amount: float) -> Dict:
    """Declare a dividend distribution to all shareholders.

    Distributed pro-rata based on ownership. Can only distribute from
    retained earnings (not invested capital).

    Args:
        amount: Total dividend amount ($).

    Returns:
        Dict with distribution details.
    """
    return _client.call('declare_dividend', {'amount': amount})
