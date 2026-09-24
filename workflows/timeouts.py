"""Activity time budgets for Temporal workflows.

Book-length docs (thousands of translation blocks / hundreds of OCR pages)
legitimately run for many hours. Temporal still requires a finite
``start_to_close`` — use a multi-day budget as "effectively unlimited".

Heartbeat applies only after PENDING → RUNNING (Temporal activity started).
Queued Postgres waiters never start an activity, so they are never heartbeated.
``HEARTBEAT`` is a process-liveness lease (≫ the ~20s ping in ``_common``),
not a progress deadline — stall kills stay off.
"""

from datetime import timedelta

# Effectively unlimited for user-facing long work (translate / OCR / digest).
LONG_RUN = timedelta(days=14)

# RUNNING activities only: detect a dead worker without waiting out LONG_RUN.
HEARTBEAT = timedelta(minutes=15)

# fail_* / small bookkeeping activities only.
BOOKKEEPING = timedelta(minutes=10)

# Wait-gates that poll "is OCR done yet?" — short attempt, unlimited retries.
WAIT_GATE = timedelta(minutes=5)
