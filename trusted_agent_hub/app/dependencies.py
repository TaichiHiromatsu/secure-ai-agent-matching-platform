from fastapi.templating import Jinja2Templates

# Shared Jinja2 template loader
# Note: other routers can import this to avoid duplicating template initialization.
templates = Jinja2Templates(directory="app/templates")

__all__ = ["templates"]
