"""Reporter framework of the experiment package (the T7 component).

The reporters consume one finished
:class:`~configurable_automl_engine.experiment.result.ExperimentResult`
and produce artifacts: a JSON snapshot (:class:`JSONReporter`), a
human-readable Markdown document (:class:`MarkdownReporter`) or any
custom sink built on the :class:`Reporter` interface. Several reporters
run in one call through :class:`CompositeReporter` — the single
registration point of a run.

The reporters are pure consumers of the result model: they never import
the training engine and never import MLflow (an MLflow reporter is a
separate adapter, outside this package).
"""

from configurable_automl_engine.experiment.reporting.base import (
    CompositeReporter,
    Reporter,
    ReporterArtifact,
)
from configurable_automl_engine.experiment.reporting.json_reporter import (
    JSONReporter,
)
from configurable_automl_engine.experiment.reporting.markdown_reporter import (
    MarkdownReporter,
)

__all__ = [
    "CompositeReporter",
    "JSONReporter",
    "MarkdownReporter",
    "Reporter",
    "ReporterArtifact",
]
