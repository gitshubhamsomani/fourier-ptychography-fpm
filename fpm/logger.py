"""
Copyright (c) 2026, John Meshreki
All rights reserved.

john.meshreki@gmail.com

-----------------------------------------------------
Shared logger and run-output manager for the FPM codebase.  
"""


from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class FPMLoggerObject:
    """
    Shared logger and run-output manager for the FPM codebase.

    This class provides:
    - one shared class-level logger
    - console logging to stderr
    - optional file logging
    - automatic creation of a unique run directory under a chosen output root
    - automatic creation of a metadata subdirectory
    - saving of the experiment JSON into metadata

    Directory layout
    ----------------
    If the output root is, for example, "output", then each run creates:

        output/
          output_YYYY_MM_DD_HH_MM_SS/
            metadata/
              run.log
              experiment_config.json

    Typical usage
    -------------
    1. At the start of the run:
        FPMLoggerObject.set_debug_level(logging.INFO)
        run_dir = FPMLoggerObject.create_timestamped_output_dir("output")
        FPMLoggerObject.set_log_file_path_from_output_dir()
        FPMLoggerObject.save_json_config(cfg.model_dump())

    2. Replace print(...) with:
        FPMLoggerObject.log.info("Finished loading images")
        FPMLoggerObject.log.debug("N_obj: %s", N_obj)
        FPMLoggerObject.log.warning("Missing file: %s", path)

    Notes
    -----
    - The logger is initialized once at class definition time.
    - File logging is appended to `metadata/run.log`.
    - The generated run directory name uses local time and the format:
      output_YYYY_MM_DD_HH_MM_SS
    """

    log: logging.Logger = logging.getLogger("fpm")
    level: int = logging.INFO
    log_file_path: str | None = None
    output_root: str | None = None
    output_dir: str | None = None
    metadata_dir: str | None = None
    initialized: bool = False

    if not initialized:
        log.setLevel(level)

        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setLevel(level)
        formatter = logging.Formatter(
            "%(levelname)s:%(name)s:%(module)s:%(lineno)s -- %(asctime)s -- %(message)s"
        )
        stream_handler.setFormatter(formatter)
        log.addHandler(stream_handler)

        initialized = True

    @classmethod
    def set_debug_level(cls, level: int) -> None:
        """
        Set the log level for the shared logger and all handlers.

        Parameters
        ----------
        level:
            Logging level such as logging.DEBUG or logging.INFO.
        """
        cls.level = level
        cls.log.setLevel(level)
        for handler in cls.log.handlers:
            handler.setLevel(level)

    @classmethod
    def create_timestamped_output_dir(
        cls,
        output_root: str | Path = "output",
    ) -> str:
        """
        Create a new timestamped run directory and its metadata subdirectory.

        Parameters
        ----------
        output_root:
            Root directory under which run directories are created.
        prefix:
            Prefix for the run directory name.

        Returns
        -------
        str
            Full path to the created run directory.

        Example
        -------
        If called at 2026-03-27 15:52:55, this creates:

            output/2026_03_27/output_2026_03_27_15_52_55/
            output/2026_03_27/output_2026_03_27_15_52_55/metadata/
        """
        now = datetime.now()
        day_folder = now.strftime("%Y_%m_%d")
        timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
        run_name = f"{timestamp}"

        output_root = Path(output_root) / day_folder
        output_root.mkdir(parents=True, exist_ok=True)
        run_dir = output_root / run_name
        metadata_dir = run_dir / "metadata"

        metadata_dir.mkdir(parents=True, exist_ok=True)

        cls.output_root = str(output_root)
        cls.output_dir = str(run_dir)
        cls.metadata_dir = str(metadata_dir)

        cls.log.info("Created run directory: %s", run_dir)
        cls.log.info("Created metadata directory: %s", metadata_dir)

        return str(run_dir)

    @classmethod
    def set_output_dir(cls, output_dir: str | Path) -> None:
        """
        Use an existing output directory and ensure metadata exists.

        Parameters
        ----------
        output_dir:
            Output directory for the run.
        """
        output_dir = Path(output_dir)
        metadata_dir = output_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        cls.output_dir = str(output_dir)
        cls.metadata_dir = str(metadata_dir)

    @classmethod
    def set_log_file_path(cls, path: str | Path) -> None:
        """
        Attach a file handler that writes logs to the given path.

        Any existing file handler is removed before the new one is added.

        Parameters
        ----------
        path:
            Path to the log file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        cls.log_file_path = str(path)

        for handler in list(cls.log.handlers):
            if isinstance(handler, logging.FileHandler):
                handler.close()
                cls.log.removeHandler(handler)

        file_handler = logging.FileHandler(cls.log_file_path, mode="a", encoding="utf-8")
        file_handler.setLevel(cls.level)
        formatter = logging.Formatter(
            "%(levelname)s:%(name)s:%(module)s:%(lineno)s -- %(asctime)s -- %(message)s"
        )
        file_handler.setFormatter(formatter)
        cls.log.addHandler(file_handler)

        cls.log.info("Attached file logger: %s", cls.log_file_path)

    @classmethod
    def set_log_file_path_from_output_dir(cls, filename: str = "run.log") -> None:
        """
        Create/use the log file under the metadata directory.

        Parameters
        ----------
        filename:
            Log filename, default is 'run.log'.

        Raises
        ------
        ValueError
            If no output directory was configured first.
        """
        if cls.metadata_dir is None:
            raise ValueError("Output directory is not set. Call create_timestamped_output_dir(...) first.")

        cls.set_log_file_path(Path(cls.metadata_dir) / filename)

    @classmethod
    def save_json_config(
        cls,
        config: dict[str, Any],
        filename: str = "experiment_config.json",
    ) -> str:
        """
        Save a JSON configuration snapshot into the metadata directory.

        Parameters
        ----------
        config:
            Dictionary to serialize as JSON.
        filename:
            Output filename inside metadata.

        Returns
        -------
        str
            Full path to the saved JSON file.

        Raises
        ------
        ValueError
            If no output directory was configured first.
        """
        if cls.metadata_dir is None:
            raise ValueError("Output directory is not set. Call create_timestamped_output_dir(...) first.")

        out_path = Path(cls.metadata_dir) / filename
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        cls.log.info("Saved config JSON to %s", out_path)
        return str(out_path)

    @classmethod
    def save_text_metadata(cls, text: str, filename: str = "run_notes.txt") -> str:
        """
        Save a small text metadata file into the metadata directory.

        Parameters
        ----------
        text:
            Text content to save.
        filename:
            Output text filename.

        Returns
        -------
        str
            Full path to the saved text file.
        """
        if cls.metadata_dir is None:
            raise ValueError("Output directory is not set. Call create_timestamped_output_dir(...) first.")

        out_path = Path(cls.metadata_dir) / filename
        with out_path.open("w", encoding="utf-8") as f:
            f.write(text)

        cls.log.info("Saved text metadata to %s", out_path)
        return str(out_path)