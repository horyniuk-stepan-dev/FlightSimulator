# Skill Observation Log

Observations captured during task-oriented work. Each entry identifies a
potential skill improvement or new skill opportunity.

**Status key:** OPEN = not yet actioned | ACTIONED = skill updated/created |
DECLINED = user decided not to pursue

---

## 2026-09-18 — Simulator dataset validation

### Observation 1: Separate evaluator truth structurally from system inputs

**Status:** OPEN
**Date:** 2026-09-18
**Session context:** Implementing a reproducible video benchmark generator with reference and query runs.
**Skill:** New skill candidate: synthetic-benchmark-dataset
**Type:** open-source
**Phase/Area:** Dataset layout and leakage prevention

**Issue:** A batch generator initially emitted calibration and slot-level truth
for query runs as well as reference runs. Even when documentation says not to
use query truth, placing input-compatible truth beside a query makes accidental
evaluation leakage easy. Count consistency also required explicit checks across
decoded video, per-frame truth, telemetry and slot-level exports.

**Suggested improvement:** Create a reusable benchmark-generation skill that
requires role-specific output schemas: reference runs may contain calibration,
while query runs expose evaluator truth only in a distinct artifact. Require an
atomic manifest to validate decoded frame count, contiguous frame IDs, fixed-time
timestamps, file hashes and completion status before accepting a run.

**Principle:** Prevent evaluation leakage with directory and schema design, then
enforce temporal and file identity mechanically; prose warnings alone are too
easy to bypass accidentally.
