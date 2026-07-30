"""Route tests for admin user management (TDD step 5)."""

from __future__ import annotations

from src.extensions import db
from src.models import User
from src.services.auth_service import hash_password


def _login_as_admin(client: object, app: object) -> None:
    with app.app_context():
        if db.session.query(User).filter_by(username="admin_test").count() == 0:
            db.session.add(
                User(username="admin_test", password_hash=hash_password("adminpw"), role="admin")
            )
            db.session.commit()
    client.post("/login", data={"username": "admin_test", "password": "adminpw"})


def _login_as_user(client: object, app: object, username: str = "regular") -> None:
    with app.app_context():
        if db.session.query(User).filter_by(username=username).count() == 0:
            db.session.add(User(username=username, password_hash=hash_password("pw")))
            db.session.commit()
    client.post("/login", data={"username": username, "password": "pw"})


class TestAdminListUsers:
    def test_admin_can_list(self, client: object, app: object) -> None:
        _login_as_admin(client, app)
        res = client.get("/admin/users")
        assert res.status_code == 200
        data = res.get_json()
        assert "users" in data
        assert isinstance(data["users"], list)

    def test_non_admin_gets_403(self, client: object, app: object) -> None:
        _login_as_user(client, app, "regular1")
        res = client.get("/admin/users")
        assert res.status_code == 403

    def test_unauthenticated_redirects(self, client: object) -> None:
        res = client.get("/admin/users", follow_redirects=False)
        assert res.status_code in (301, 302, 303)


class TestAdminDisableUser:
    def test_admin_can_disable(self, client: object, app: object) -> None:
        _login_as_admin(client, app)
        with app.app_context():
            u = User(username="todisable", password_hash=hash_password("pw"), role="user")
            db.session.add(u)
            db.session.commit()
            uid = u.id
        res = client.post(f"/admin/users/{uid}/disable", follow_redirects=True)
        assert res.status_code == 200
        with app.app_context():
            u = db.session.get(User, uid)
            assert u.role == "disabled"

    def test_cannot_disable_admin(self, client: object, app: object) -> None:
        _login_as_admin(client, app)
        with app.app_context():
            admin = db.session.query(User).filter_by(username="admin_test").first()
            assert admin is not None
            res = client.post(f"/admin/users/{admin.id}/disable", follow_redirects=True)
            assert res.status_code == 200
            with app.app_context():
                admin = db.session.get(User, admin.id)
                assert admin.role == "admin"  # unchanged

    def test_non_admin_gets_403(self, client: object, app: object) -> None:
        _login_as_user(client, app, "regular2")
        res = client.post("/admin/users/9999/disable")
        assert res.status_code == 403


class TestAdminEnableUser:
    def test_admin_can_enable(self, client: object, app: object) -> None:
        _login_as_admin(client, app)
        with app.app_context():
            u = User(username="toenable", password_hash=hash_password("pw"), role="disabled")
            db.session.add(u)
            db.session.commit()
            uid = u.id
        res = client.post(f"/admin/users/{uid}/enable", follow_redirects=True)
        assert res.status_code == 200
        with app.app_context():
            u = db.session.get(User, uid)
            assert u.role == "user"

    def test_cannot_enable_non_disabled(self, client: object, app: object) -> None:
        _login_as_admin(client, app)
        with app.app_context():
            u = User(username="activeuser", password_hash=hash_password("pw"), role="user")
            db.session.add(u)
            db.session.commit()
            res = client.post(f"/admin/users/{u.id}/enable", follow_redirects=True)
            assert res.status_code == 200
            with app.app_context():
                u = db.session.get(User, u.id)
                assert u.role == "user"  # unchanged
