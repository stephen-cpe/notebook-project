"""Flask extensions instantiated once and bound to the app in the factory.

Centralizing the extension objects avoids circular imports between the
``src`` package and the route/service modules.
"""

from __future__ import annotations

from flask_login import LoginManager
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
