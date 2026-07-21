# Proxy WSGI entrypoint for Render and cloud deployments targeting app:app
from app_web import app

if __name__ == '__main__':
    app.run()
