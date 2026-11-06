"""Reporter framework of the experiment package (the T7 component).

The reporters consume one finished
:class:`~configurable_automl_engine.experiment.result.ExperimentResult`
and produce artifacts: a JSON snapshot (:class:`JSONReporter`), a
human-readable Markdown document (:class:`MarkdownReporter`) or any
custom sink built on the :class:`Reporter` interface. Several reporters
run in one call through :class:`CompositeReporter` — the single
registration point of a run.

The reporters are pure consumers of the result model: they never import
the training engine and never import MLflow. The one exception is the
MLflow *adapter* (:class:`MLflowReporter`), which imports ``mlflow``
lazily, inside its own ``report`` call — importing this package stays
MLflow-free.
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
from configurable_automl_engine.experiment.reporting.mlflow_reporter import (
    MLflowReporter,
    MLflowReporterError,
)

__all__ = [
    "CompositeReporter",
    "JSONReporter",
    "MLflowReporter",
    "MLflowReporterError",
    "MarkdownReporter",
    "Reporter",
    "ReporterArtifact",
]
