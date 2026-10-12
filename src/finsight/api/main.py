"""ASGI entrypoint: `uvicorn finsight.api.main:app`."""

from finsight.api.app import create_app

app = create_app()
