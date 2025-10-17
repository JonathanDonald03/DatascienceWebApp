import os

# Import the Flask app instance from app.py
from app import app

# Expose 'app' for Gunicorn: `gunicorn -c gunicorn.conf.py wsgi:app`
# Also allow running directly for local testing: `python wsgi.py`
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
