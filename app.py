"""
FraudGuard AI — Root Application Entry Point
============================================

Provides the ASGI `app` instance for production servers (Uvicorn, Gunicorn, Hypercorn)
and Render Web Service deployments.
"""

import os
import uvicorn
from src.api.fraud_api import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")

