from dataclasses import dataclass

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session


@dataclass
class ItemRow:
    item_id: int
    category_id1: int
    category_id2: int | None
    brand_id: int
    price: float


def get_item_or_404(item_id: int, db: Session) -> ItemRow:
    row = db.execute(
        text("SELECT item_id, category_id1, category_id2, brand_id, price FROM item WHERE item_id = :iid"),
        {"iid": item_id},
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"item {item_id} not found")
    return ItemRow(
        item_id=row[0],
        category_id1=row[1],
        category_id2=row[2],
        brand_id=row[3],
        price=float(row[4]),
    )
