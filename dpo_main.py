"""Public DPO training entry point.

The implementation remains in :mod:`dpo.main`; this file only gives the
repository a conventional root-level command.
"""

from dpo.main import main


if __name__ == "__main__":
    raise SystemExit(main())
