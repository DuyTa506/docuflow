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


def test_all_translation_activities_are_registered_with_worker():
    from workers.temporal_worker import TRANSLATION_ACTIVITIES
    from workflows.activities import translation_activities

    defined = _activity_defn_names(translation_activities)
    registered = _activity_defn_names_from_list(TRANSLATION_ACTIVITIES)
    missing = defined - registered
    assert not missing, (
        f"Activities defined in translation_activities.py but missing from "
        f"TRANSLATION_ACTIVITIES (workflow would hang forever): {missing}"
    )


def test_fail_translation_activity_is_registered():
    from workers.temporal_worker import TRANSLATION_ACTIVITIES
    from workflows.activities.translation_activities import fail_translation_activity

    assert fail_translation_activity in TRANSLATION_ACTIVITIES


def test_all_extraction_activities_are_registered_with_worker():
    from workers.temporal_worker import EXTRACTION_ACTIVITIES
    from workflows.activities import extraction_activities

    defined = _activity_defn_names(extraction_activities)
    registered = _activity_defn_names_from_list(EXTRACTION_ACTIVITIES)
    missing = defined - registered
    assert not missing, (
        f"Activities defined in extraction_activities.py but missing from "
        f"EXTRACTION_ACTIVITIES (workflow would hang forever): {missing}"
    )
