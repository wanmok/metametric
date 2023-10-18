"""Metric definitions for semantic parsing tasks."""
import metametric.core.dsl as mm
from metametric.structures.amr import AMR

s_match = mm.normalize["f1"](mm.auto[AMR])
