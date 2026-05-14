from __future__ import annotations

try:
    from .train_paper import main
except ImportError:
    from train_paper import main


if __name__ == "__main__":
    main()
