"""Index filings from the command line: python scripts/ingest.py AAPL MSFT --count 2"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from filinglens.ingest import ingest_ticker  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--count", type=int, default=1, help="how many recent 10-Ks per company")
    parser.add_argument("--force", action="store_true", help="re-index existing filings")
    args = parser.parse_args()

    for ticker in args.tickers:
        for result in ingest_ticker(ticker, count=args.count, force=args.force):
            print(result)


if __name__ == "__main__":
    main()
