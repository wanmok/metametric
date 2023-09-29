"""Metric definitions for semantic parsing tasks."""
import autometric.core.dsl as am
from autometric.structures.amr import AMR


s_match = am.normalize["f1"](AMR.metric)
