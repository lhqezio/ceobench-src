"""SaaS Bench - A benchmark for AI agents running subscription services."""

__version__ = "0.1.0"

# Core environment
from .environment import SaaSBenchEnv, Action, StepResult, build_daily_dashboard

# Agent base class
from .agents import BaseAgent

# Configuration
from .config import BenchmarkConfig

__all__ = [
    'SaaSBenchEnv',
    'Action',
    'StepResult',
    'BaseAgent',
    'BenchmarkConfig',
    'build_daily_dashboard',
]
