"""JWT creation, current-user resolution, and FastAPI role dependencies."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt import InvalidTokenError

from .config import Settings, get_settings
from .db import Database
from .schemas import Role, UserPrincipal

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


def create_access_token(user: UserPrincipal, settings: Settings) -> str:
    """Sign a short-lived token carrying the database user ID and role."""
    expires_at = datetime.now(UTC) + timedelta(minutes=settings.access_token_minutes)
    return jwt.encode(
        {"sub": user.id, "username": user.username, "role": user.role.value, "exp": expires_at},
        settings.jwt_secret,
        algorithm="HS256",
    )


def get_current_user(
    token: str = Depends(oauth2_scheme), settings: Settings = Depends(get_settings)
) -> UserPrincipal:
    """Validate the token and reload the user so deleted accounts cannot keep access."""
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
        user_id = payload.get("sub")
        if not user_id:
            raise credentials_error
    except InvalidTokenError as exc:
        raise credentials_error from exc
    user = Database(settings).get_user(user_id)
    if not user:
        raise credentials_error
    return user


def require_roles(*roles: Role) -> Callable[[UserPrincipal], UserPrincipal]:
    """Create a route dependency that rejects users outside the allowed roles."""
    def dependency(user: UserPrincipal = Depends(get_current_user)) -> UserPrincipal:
        if user.role not in roles:
            raise HTTPException(status_code=403, detail="Insufficient role")
        return user

    return dependency
