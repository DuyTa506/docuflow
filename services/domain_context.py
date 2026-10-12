"""What every digest stage knows about its document's specialty and terminology.

``DomainProfileService`` writes it; the other stages read it here and put
``clause()`` in their prompts. A document without a profile (the stage failed,
or it predates it) gives an empty clause, so the stages behave exactly as before.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict

from data.database import get_db_manager
from utils.domain_profile import domain_clause, termbase_glossary
from utils.glossary import glossary_clause

logger = logging.getLogger(__name__)

PROMPT_MAX_TERMS = 40


@dataclass
class DomainContext:
    profile: Dict[str, Any] = field(default_factory=dict)
    termbase: Dict[str, Any] = field(default_factory=dict)

    def clause(self, text: str = "") -> str:
        """Domain line plus the termbase entries that occur in ``text`` (all of them if no text)."""
        glossary = termbase_glossary(self.termbase)
        terms = glossary_clause(glossary, text, PROMPT_MAX_TERMS) if text else ""
        if not text and glossary:
            terms = glossary_clause(glossary, " ".join(glossary), PROMPT_MAX_TERMS)
        return f"{domain_clause(self.profile)}{terms}"


def load_domain_context(document_id: str) -> DomainContext:
    try:
        from data.db_models import Document

        with get_db_manager().session() as db:
            row = db.query(Document).filter(Document.id == document_id).first()
            if row is None:
                return DomainContext()
            return DomainContext(row.domain_profile or {}, row.termbase or {})
    except Exception as exc:  # the profile only improves wording; never block a stage on it
        logger.warning("domain context unavailable for %s: %s", document_id, exc)
        return DomainContext()
