"""Workspace-local re-export of the framework TimeSeriesFromObservables chart.

The tumor studies address this chart as ``local:TimeSeriesFromObservables``.
``vivarium_tyssue.core.build_core`` registers the framework class under that
name at runtime, so it renders correctly — but the pre-publication report
linter (``pbg_superpowers.report_linter``) resolves ``local:`` addresses by a
STATIC scan of ``class`` definitions under ``<package>/visualizations/`` (no
import). Defining the subclass here (a) makes ``local:TimeSeriesFromObservables``
statically resolvable for the linter and (b) gives the workspace one obvious
place that "owns" the chart. Behaviour is identical to the base class.
"""
from pbg_superpowers.visualizations.timeseries_from_observables import (
    TimeSeriesFromObservables as _BaseTimeSeriesFromObservables,
)


class TimeSeriesFromObservables(_BaseTimeSeriesFromObservables):
    """Identical to the pbg-superpowers chart; subclassed only so the address
    ``local:TimeSeriesFromObservables`` resolves under this workspace package."""
