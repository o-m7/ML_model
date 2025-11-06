"""Main CLI entry point."""

import sys

if __name__ == '__main__':
    # Allow running as: python -m intraday_system.cli.train
    from .train import main
    main()

