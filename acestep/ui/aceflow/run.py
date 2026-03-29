"""
AceFlow v1.0
Built on top of Ace-Step v1.5

Copyright (C) 2026 Marco Robustini [Marcopter]

This file is part of AceFlow.
AceFlow is licensed under the GNU General Public License v3.0 or later.

You may redistribute and/or modify this software under the terms
of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or any later version.

AceFlow is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.
"""

"""Command-line entrypoint for serving the AceFlow ASGI application locally."""

import argparse

import uvicorn

from .app import create_app


def main():
    """Parse CLI options, create the AceFlow app, and run the Uvicorn server.

    Args:
        None: Arguments are read from the process command line.

    Returns:
        None: This function blocks until the Uvicorn server stops.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7861)
    args = ap.parse_args()

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
