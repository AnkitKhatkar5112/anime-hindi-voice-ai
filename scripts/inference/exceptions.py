"""
Custom exceptions for the production dubbing pipeline.
"""

from __future__ import annotations

from typing import List


class PipelineResumeError(Exception):
    """Raised when a pipeline stage cannot resume due to missing input files.

    Parameters
    ----------
    missing_files:
        List of file paths that were expected on disk but not found.
    """

    def __init__(self, missing_files: List[str]) -> None:
        self.missing_files = missing_files
        files_str = "\n  ".join(missing_files)
        super().__init__(
            f"Cannot resume pipeline — {len(missing_files)} required file(s) missing:\n  {files_str}"
        )


class PipelineValidationError(Exception):
    """Raised when output validation checks fail.

    Parameters
    ----------
    failed_checks:
        List of check names (or descriptive strings) that did not pass.
    """

    def __init__(self, failed_checks: List[str]) -> None:
        self.failed_checks = failed_checks
        checks_str = ", ".join(failed_checks)
        super().__init__(
            f"Output validation failed — {len(failed_checks)} check(s) did not pass: {checks_str}"
        )
