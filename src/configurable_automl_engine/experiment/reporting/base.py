"""Reporter abstraction and the composite of reporters (the T7
foundation).

A reporter consumes one finished
:class:`~configurable_automl_engine.experiment.result.ExperimentResult`
and produces an artifact: a file path or a rendered string. Reporters are
pure consumers of the result model — they never reach into the training
engine and never import MLflow (the optional MLflow adapter lives in
:mod:`.mlflow_reporter` and imports ``mlflow`` lazily, inside its own
``report`` call).

Contract:

* :class:`Reporter` is the minimal interface every reporter implements:
  one method, ``report(result)``, returning the artifact (a ``Path`` for
  file reporters, a ``str`` for text reporters). The class is generic in
  the artifact type: a plain :class:`Reporter` is
  ``Reporter[ReporterArtifact]`` (one artifact of any shape), while
  :class:`CompositeReporter` is a ``Reporter[tuple[ReporterArtifact,
  ...]]`` — so a caller with a typed ``Reporter[...]`` reference knows
  exactly what ``report`` returns, without a cast;
* :class:`CompositeReporter` runs several reporters in one call (the
  single registration point of a run: register once, report to every
  sink). It returns the tuple of the individual artifacts; a reporter
  failure propagates (fail fast — a silent half-report is worse than an
  explicit error), already-produced artifacts are kept on disk.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeAlias, TypeVar

from configurable_automl_engine.experiment.result import ExperimentResult

__all__ = [
    "CompositeReporter",
    "Reporter",
    "ReporterArtifact",
]

#: The artifact a reporter returns: a file path (file reporters) or a
#: rendered text (string reporters).
ReporterArtifact: TypeAlias = "Path | str"


#: The artifact type produced by one ``report`` call. Covariant and
#: unbounded: a concrete reporter returns one artifact (``Reporter[Path]``,
#: ``Reporter[str]``), the composite returns a tuple of them
#: (``Reporter[tuple[ReporterArtifact, ...]]``) — a tuple of artifacts is
#: not itself an artifact, so no bound is possible.
_ArtifactT_co = TypeVar("_ArtifactT_co", covariant=True)


class Reporter(ABC, Generic[_ArtifactT_co]):
    """Abstract base of every experiment reporter.

    Subclasses implement :meth:`report`, consuming a finished
    ``ExperimentResult`` and returning the produced artifact. Reporters
    are stateless consumers of the result model: they never import the
    training engine and never import MLflow (the optional MLflow adapter
    imports ``mlflow`` lazily, inside its own ``report`` call).

    A plain reporter returns one artifact; a composite returns a tuple
    (see :class:`CompositeReporter`), and the generic parameter makes
    the difference visible to the type checker.
    """

    @abstractmethod
    def report(self, result: ExperimentResult) -> _ArtifactT_co:
        """Render the experiment result into the reporter's artifact.

        Args:
            result: The finished experiment result (the single input of
                every reporter; the model already holds the config
                snapshot, the run records and the scores).

        Returns:
            The produced artifact: the written file path for a file
            reporter, the rendered text for a string reporter, the tuple
            of the individual artifacts for a composite.

        Raises:
            Exception: Whatever the reporter's sink raises (an
                unwritable destination, a serialization error); the
                exception propagates to the caller.
        """


class CompositeReporter(Reporter[tuple[ReporterArtifact, ...]]):
    """Run several reporters in one call.

    The composite is the single registration point of a run: register
    every sink once, then one ``report`` call produces every artifact
    with the same ``ExperimentResult``. The reporters are invoked in
    registration order; each returns its own artifact.

    Args:
        reporters: The reporters to run, in invocation order. Empty is
            allowed (the composite is a no-op returning an empty tuple).
    """

    def __init__(
        self, reporters: list[Reporter[Any]] | tuple[Reporter[Any], ...]
    ) -> None:
        self._reporters: tuple[Reporter[Any], ...] = tuple(reporters)

    @property
    def reporters(self) -> tuple[Reporter[Any], ...]:
        """The registered reporters, in invocation order."""
        return self._reporters

    def report(self, result: ExperimentResult) -> tuple[ReporterArtifact, ...]:
        """Run every registered reporter on the result, in order.

        Args:
            result: The finished experiment result passed to every
                reporter.

        Returns:
            tuple[ReporterArtifact, ...]: The artifacts of the
                individual reporters, in registration order (empty for a
                composite without reporters).

        Raises:
            Exception: The first reporter failure propagates (fail
                fast); artifacts of the already-run reporters stay
                produced.
        """
        return tuple(reporter.report(result) for reporter in self._reporters)
