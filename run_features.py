"""Developer entrypoint: transform + derive tables only."""

from __future__ import annotations

from data_engineering import derive_tables_local, transform_data_local


def main() -> None:
    transform_data_local.main()
    derive_tables_local.main()


if __name__ == "__main__":
    main()
