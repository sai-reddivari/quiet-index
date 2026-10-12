"""Serve the dashboard on loopback only; no external publication."""

import argparse

from quiet_index.demo_web import serve


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8790)
    serve(parser.parse_args().port)
