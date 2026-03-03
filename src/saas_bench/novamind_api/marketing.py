"""Marketing and advertising tools."""

from typing import Dict, Optional
from . import _client


def set_daily_spend(advertising: Optional[float] = None,
                    operations: Optional[float] = None,
                    development: Optional[float] = None) -> Dict:
    """Set daily spending for advertising, operations, and development.

    Args:
        advertising: Daily advertising budget ($).
        operations: Daily operations budget ($).
        development: Daily development budget ($).

    Returns:
        Dict with update confirmation.
    """
    args = {}
    if advertising is not None:
        args['advertising'] = advertising
    if operations is not None:
        args['operations'] = operations
    if development is not None:
        args['development'] = development
    return _client.call('set_daily_spend', args)


def set_ad_channel_spend(**channel_percentages: float) -> Dict:
    """Set per-channel advertising budget allocation as percentages.

    Args:
        social_media: Percentage allocation (0.0-1.0).
        search_ads: Percentage allocation (0.0-1.0).
        linkedin: Percentage allocation (0.0-1.0).
        content_marketing: Percentage allocation (0.0-1.0).
        referral_program: Percentage allocation (0.0-1.0).

    Returns:
        Dict with update confirmation.
    """
    return _client.call('set_ad_channel_spend', channel_percentages)


def set_targeted_ad_spend(targeted_spend: Dict[str, Dict[str, float]]) -> Dict:
    """Set per-group targeted advertising spend.

    Args:
        targeted_spend: {group_id: {channel: amount}} nested dict.

    Returns:
        Dict with update confirmation.
    """
    return _client.call('set_targeted_ad_spend', {'targeted_spend': targeted_spend})


def set_ads_strength(global_strength: Optional[float] = None,
                     by_group: Optional[Dict[str, float]] = None,
                     by_customer: Optional[Dict[int, float]] = None) -> Dict:
    """Set advertising strength multipliers.

    Args:
        global_strength: Global ad strength multiplier (default 1.0).
        by_group: Per-group overrides {group_id: strength}.
        by_customer: Per-customer overrides {customer_id: strength}.

    Returns:
        Dict with update confirmation.
    """
    args = {}
    if global_strength is not None:
        args['global_strength'] = global_strength
    if by_group is not None:
        args['by_group'] = by_group
    if by_customer is not None:
        args['by_customer'] = by_customer
    return _client.call('set_ads_strength', args)


def set_lead_promotion(global_promotion: Optional[float] = None,
                       by_group: Optional[Dict[str, float]] = None,
                       by_channel: Optional[Dict[str, float]] = None,
                       by_channel_group: Optional[Dict[str, Dict[str, float]]] = None) -> Dict:
    """Set lead acquisition promotions (first-month discounts for new leads).

    Args:
        global_promotion: Default lead promotion ($/month off first month).
        by_group: Per-group overrides {group_id: promotion}.
        by_channel: Per-channel overrides {channel: promotion}.
        by_channel_group: Per-channel-group overrides {channel: {group_id: promotion}}.

    Returns:
        Dict with update confirmation.
    """
    args = {}
    if global_promotion is not None:
        args['global_promotion'] = global_promotion
    if by_group is not None:
        args['by_group'] = by_group
    if by_channel is not None:
        args['by_channel'] = by_channel
    if by_channel_group is not None:
        args['by_channel_group'] = by_channel_group
    return _client.call('set_lead_promotion', args)
