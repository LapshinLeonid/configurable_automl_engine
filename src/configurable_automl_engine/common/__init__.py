"""Shared utilities: the bottom import layer of the package.

Definitions (``definitions``, ``settings``, ``errors``), input validation
(``validation_utils``), dependency checks, artifact serialization, scoring
metrics (``metrics``), logging setup (``log_setup``) and the shared-memory
``DataFrame`` (``shared_frame``). Modules here import nothing from the package
outside ``common`` (and the generated ``_version``). The package ``__init__``
imports nothing, so importing a single submodule stays cheap.
"""
