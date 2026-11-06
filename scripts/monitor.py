#!/usr/bin/env python3
"""
scripts/monitor.py
==================
Standalone CLI entry point for the BrainTumorXAI Health Monitoring & Keep-Alive service.

Usage:
    python scripts/monitor.py
    # or with custom env:
    APP_URL=https://brain-tumor-xai-jf4z.onrender.com CHECK_INTERVAL=300 python scripts/monitor.py
"""

import os
import signal
import sys

# Ensure project root is available on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logger import configure_logging
from src.monitor import HealthMonitor, MonitorConfig

def main():
    # Configure logging
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logger = configure_logging(level=log_level)

    print("=================================================================")
    print("  BrainTumorXAI — Standalone Render Keep-Alive Health Monitor   ")
    print("=================================================================")

    try:
        config = MonitorConfig()
    except ValueError as val_err:
        print(f"Configuration error: {val_err}", file=sys.stderr)
        sys.exit(1)

    print(f"Target URL:       {config.target_url}")
    print(f"Check Interval:   {config.check_interval}s")
    print(f"Request Timeout:  {config.request_timeout}s")
    print(f"Max Retries:      {config.max_retries}")
    print(f"Retry Base Delay: {config.retry_base_delay}s")
    print("-----------------------------------------------------------------")
    print("Starting periodic monitoring loop. Press Ctrl+C to stop.\n")

    monitor = HealthMonitor(config=config)

    # Attach graceful termination signals
    def _signal_handler(sig, frame):
        print("\nShutdown signal caught. Gracefully stopping health monitor...")
        monitor.stop()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    monitor.run_loop()
    print("Health monitor terminated cleanly.")

if __name__ == "__main__":
    main()
