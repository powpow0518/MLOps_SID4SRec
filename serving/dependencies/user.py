from dataclasses import dataclass

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session


@dataclass
class UserRow:
    user_id: int


def get_user_or_404(user_id: int, db: Session) -> UserRow:
    row = db.execute(
        text('SELECT user_id FROM "user" WHERE user_id = :uid'),
        {"uid": user_id},
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"user {user_id} not found")
    return UserRow(user_id=row[0])
