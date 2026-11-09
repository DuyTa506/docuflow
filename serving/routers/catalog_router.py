"""
CTĐT catalog used by §3 of the digest.

GET    /api/v2/catalog/ctdt   — the catalog in effect, and where it came from
PUT    /api/v2/catalog/ctdt   — replace the catalog (ADMIN)
DELETE /api/v2/catalog/ctdt   — drop the upload, revert to the bundled one (ADMIN)

The bundled catalog is Phụ lục I of Thông tư 09/2022/TT-BGDĐT filtered to the
Academy's scope; another deployment may need a different one without editing a
file in the repo. When nothing is loaded, an empty §3 is intentional —
`source: "none"` is how a caller tells that apart from "no discipline fits".
"""

from fastapi import APIRouter, Body, Depends, HTTPException

from api.dependencies import get_current_user, require_role
from api.schemas import CtdtCatalogResponse
from data.db_models import User
from utils.ctdt_catalog import (
    CATALOG_KEYS,
    catalog_source,
    clear_catalog,
    count_programmes,
    has_entries,
    load_catalog,
    override_path,
    save_catalog,
)

router = APIRouter(tags=["catalog"])


def _state() -> CtdtCatalogResponse:
    catalog = load_catalog()
    return CtdtCatalogResponse(
        source=catalog_source(),
        upload_path=str(override_path()),
        has_entries=has_entries(catalog),
        # Count disciplines (leaves), not groups — a group is only a heading.
        counts={k: count_programmes(catalog, k) for k in CATALOG_KEYS},
        catalog=catalog,
    )


@router.get("/api/v2/catalog/ctdt", response_model=CtdtCatalogResponse)
async def get_ctdt_catalog(_user: User = Depends(get_current_user)):
    """Read the CTĐT catalog currently in effect."""
    return _state()


@router.put("/api/v2/catalog/ctdt", response_model=CtdtCatalogResponse)
async def put_ctdt_catalog(
    catalog: dict = Body(..., description="Object keyed by undergraduate/master/phd/…"),
    _admin: User = Depends(require_role("ADMIN")),
):
    """Replace the CTĐT catalog (ADMIN). The bundled file is never modified."""
    try:
        save_catalog(catalog)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _state()


@router.delete("/api/v2/catalog/ctdt", response_model=CtdtCatalogResponse)
async def delete_ctdt_catalog(_admin: User = Depends(require_role("ADMIN"))):
    """Remove the uploaded catalog so the bundled one applies again (ADMIN)."""
    clear_catalog()
    return _state()
