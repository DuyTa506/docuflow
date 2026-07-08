"""Every activity defined for the digest pipeline must be registered with the
Temporal worker — otherwise the workflow hangs forever waiting on an activity
no worker can ever execute (see workers/temporal_worker.py PIPELINE_ACTIVITIES).
"""
import inspect

from workers.temporal_worker import PIPELINE_ACTIVITIES
from workflows.activities import digest_activities


def _activity_defn_names(module) -> set[str]:
    names = set()
    for _, obj in inspect.getmembers(module):
        defn = getattr(obj, "__temporal_activity_definition", None)
        if defn is not None:
            names.add(defn.name)
    return names


def test_all_defined_activities_are_registered_with_worker():
    defined = _activity_defn_names(digest_activities)
    registered = _activity_defn_names_from_list(PIPELINE_ACTIVITIES)
    missing = defined - registered
    assert not missing, (
        f"Activities defined in digest_activities.py but missing from "
        f"PIPELINE_ACTIVITIES (worker will never execute them): {missing}"
    )


def _activity_defn_names_from_list(activities) -> set[str]:
    names = set()
    for fn in activities:
        defn = getattr(fn, "__temporal_activity_definition", None)
        assert defn is not None, f"{fn!r} is not an @activity.defn function"
        names.add(defn.name)
    return names


def test_fail_pipeline_activity_is_registered():
    assert digest_activities.fail_pipeline_activity in PIPELINE_ACTIVITIES
