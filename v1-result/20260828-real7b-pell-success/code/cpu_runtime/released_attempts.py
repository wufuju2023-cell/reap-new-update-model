"""Same-process latest-published reservation; source preparation stays outside locks.

The provider must share the actual owning learner object with publication.
Cross-process readers of release_head are not interchangeable with this guard.
"""
from copy import deepcopy
from gpu_runtime.learner_release_store import canonical_bytes, content_sha256
from gpu_runtime.release_head import available_head
from . import matchmaker as mm


def validate_selection(value, intent, run_sha):
    mm._fields(value, "schema_version intent_sha256 learner_run_sha256 head_sha256 head", "publication selection")
    mm.require(value["schema_version"] == "reap.attempt.publication-selection.v1"
        and value["intent_sha256"] == intent["intent_sha256"] and value["learner_run_sha256"] == run_sha,
        "publication selection identity differs")
    head = value["head"]
    mm._fields(head, "schema_version run_sha256 step checkpoint_sha256 previous_head_sha256 publication_intent_sha256 model_release_sha256", "publication head")
    mm.require(head["schema_version"] == "reap.learner.release-head.v1" and head["run_sha256"] == run_sha
        and content_sha256(head) == value["head_sha256"]
        and head["model_release_sha256"] == intent["proposal"]["model_release_sha256"], "publication selection pin differs")
    mm._int(head["step"], 1, 10**10, "published step")
    for name in ("run_sha256", "checkpoint_sha256", "publication_intent_sha256", "model_release_sha256"):
        mm._sha(head[name], name)
    if head["previous_head_sha256"] is not None:
        mm._sha(head["previous_head_sha256"], "previous head")
    publication_intent = {k: v for k, v in head.items() if k not in ("publication_intent_sha256", "model_release_sha256")}
    mm.require(content_sha256(publication_intent) == head["publication_intent_sha256"], "publication intent checksum differs")


class ReleasedAttemptProvider:
    def __init__(self, learner):
        self.learner = learner
        self.identity = {"schema_version": "reap.same-process-release-provider.v1", "learner_run_sha256": learner.run_sha,
                         "atomicity": "owning-coordinator-RLock-and-scheduler-reserve"}

    def __call__(self, scheduler, prepare_source, should_stop=lambda: False):
        learner = self.learner
        # Pure exhaustion check: the placeholder is never reserved or emitted.
        # Completed replay must not require a live publisher or invoke callbacks.
        if scheduler.plan_next("0"*64) is None:
            return None
        with learner.lock:
            head = available_head(learner)
            proposal = scheduler.plan_next(head["model_release_sha256"])
        if proposal is None or should_stop():
            return None
        # Exactly one preparation. Its problem/polarity/budget cannot be changed
        # if publishing or another scheduler action happens during preparation.
        source = prepare_source(deepcopy(proposal))
        with learner.lock:
            if should_stop():
                return None
            head = available_head(learner)
            current = scheduler.plan_next(head["model_release_sha256"])
            without_release = lambda p: None if p is None else {k: v for k, v in p.items() if k != "model_release_sha256"}
            mm.require(canonical_bytes(without_release(current)) == canonical_bytes(without_release(proposal)),
                       "scheduler changed during source preparation; do not prepare or dispatch twice")
            intent = scheduler.reserve(current, source)
            selection = {"schema_version": "reap.attempt.publication-selection.v1", "intent_sha256": intent["intent_sha256"],
                "learner_run_sha256": learner.run_sha, "head_sha256": content_sha256(head), "head": head}
            validate_selection(selection, intent, learner.run_sha)
            return intent, selection
