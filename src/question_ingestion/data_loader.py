"""Data loader for CSV and Excel files with automatic statistics generation."""

from pathlib import Path
from typing import Any, Optional, Union
import re

from ..agents.models import ColumnInfo, DataSummary


class DataLoader:
    """
    Loads CSV and Excel data files and generates summary statistics.

    Provides data context that agents can use to make informed decisions
    about modeling approaches and data handling.
    """

    # Common date column name patterns
    DATE_PATTERNS = [
        r"date", r"time", r"year", r"month", r"day",
        r"timestamp", r"created", r"updated", r"period",
    ]

    # Common geographic column patterns
    GEO_PATTERNS = [
        r"state", r"city", r"country", r"region", r"county",
        r"zip", r"postal", r"location", r"address", r"lat", r"lon",
    ]

    def __init__(self):
        """Initialize the data loader."""
        self._pd = None  # Lazy load pandas

    @property
    def pd(self):
        """Lazy load pandas to avoid import errors if not installed."""
        if self._pd is None:
            try:
                import pandas as pd
                self._pd = pd
            except ImportError:
                raise ImportError(
                    "pandas is required for data loading. Install with: pip install pandas"
                )
        return self._pd

    def load_file(self, file_path: Path) -> "pd.DataFrame":
        """
        Load a single data file.

        Args:
            file_path: Path to CSV or Excel file.

        Returns:
            Pandas DataFrame with the loaded data.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".csv":
            return self.pd.read_csv(path)
        elif suffix in (".xlsx", ".xls"):
            return self.pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def load_files(self, file_paths: list[Path]) -> dict[str, "pd.DataFrame"]:
        """
        Load multiple data files.

        Args:
            file_paths: List of paths to data files.

        Returns:
            Dictionary mapping file names to DataFrames.
        """
        dataframes = {}
        for path in file_paths:
            path = Path(path)
            df = self.load_file(path)
            dataframes[path.name] = df
        return dataframes

    def analyze_column(self, series: "pd.Series") -> ColumnInfo:
        """
        Analyze a single column and generate statistics.

        Args:
            series: Pandas Series to analyze.

        Returns:
            ColumnInfo with column statistics.
        """
        pd = self.pd
        name = series.name
        total = len(series)
        missing = series.isna().sum()
        missing_pct = (missing / total * 100) if total > 0 else 0.0

        # Determine data type
        if pd.api.types.is_numeric_dtype(series):
            dtype = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            dtype = "datetime"
        elif series.dtype == "object":
            # Check if it looks like dates
            sample = series.dropna().head(10)
            if self._looks_like_dates(sample):
                dtype = "datetime"
            elif series.nunique() < len(series) * 0.5:
                dtype = "categorical"
            else:
                dtype = "text"
        else:
            dtype = "categorical"

        # Get unique count and sample values
        unique_count = series.nunique()
        sample_values = series.dropna().head(5).tolist()

        # Calculate stats for numeric columns
        stats = None
        if dtype == "numeric" and not series.isna().all():
            clean = series.dropna()
            stats = {
                "mean": float(clean.mean()),
                "std": float(clean.std()) if len(clean) > 1 else 0.0,
                "min": float(clean.min()),
                "max": float(clean.max()),
                "median": float(clean.median()),
                "q25": float(clean.quantile(0.25)),
                "q75": float(clean.quantile(0.75)),
            }

        return ColumnInfo(
            name=str(name),
            dtype=dtype,
            unique_count=unique_count,
            missing_pct=missing_pct,
            sample_values=sample_values,
            stats=stats,
        )

    def _looks_like_dates(self, sample: "pd.Series") -> bool:
        """Check if sample values look like dates."""
        pd = self.pd
        try:
            # Try to parse as dates
            parsed = pd.to_datetime(sample, errors="coerce")
            return parsed.notna().sum() > len(sample) * 0.8
        except Exception:
            return False

    def _detect_date_range(
        self,
        dataframes: dict[str, "pd.DataFrame"],
    ) -> Optional[tuple[str, str]]:
        """Detect date range from datetime columns."""
        pd = self.pd
        min_date = None
        max_date = None

        for df in dataframes.values():
            for col in df.columns:
                col_lower = str(col).lower()

                # Check if column name suggests dates
                is_date_col = any(
                    re.search(pattern, col_lower)
                    for pattern in self.DATE_PATTERNS
                )

                if not is_date_col:
                    continue

                try:
                    dates = pd.to_datetime(df[col], errors="coerce")
                    dates = dates.dropna()

                    if len(dates) > 0:
                        col_min = dates.min()
                        col_max = dates.max()

                        if min_date is None or col_min < min_date:
                            min_date = col_min
                        if max_date is None or col_max > max_date:
                            max_date = col_max
                except Exception:
                    continue

        if min_date is not None and max_date is not None:
            return (min_date.isoformat()[:10], max_date.isoformat()[:10])

        return None

    def _detect_geographic_scope(
        self,
        dataframes: dict[str, "pd.DataFrame"],
    ) -> Optional[str]:
        """Detect geographic scope from location columns."""
        locations = set()

        for df in dataframes.values():
            for col in df.columns:
                col_lower = str(col).lower()

                # Check if column name suggests geography
                is_geo_col = any(
                    re.search(pattern, col_lower)
                    for pattern in self.GEO_PATTERNS
                )

                if not is_geo_col:
                    continue

                # Get unique values
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 100:  # Reasonable number of locations
                    locations.update(str(v) for v in unique_vals[:20])

        if locations:
            sample = sorted(locations)[:5]
            if len(locations) > 5:
                return f"{', '.join(sample)}... ({len(locations)} total)"
            return ", ".join(sample)

        return None

    def _detect_data_quality_issues(
        self,
        dataframes: dict[str, "pd.DataFrame"],
    ) -> list[str]:
        """Detect common data quality issues."""
        issues = []

        for file_name, df in dataframes.items():
            # Check for high missing data
            missing_pct = df.isna().sum().sum() / df.size * 100 if df.size > 0 else 0
            if missing_pct > 20:
                issues.append(f"{file_name}: {missing_pct:.1f}% missing data")

            # Check for duplicate rows
            dup_count = df.duplicated().sum()
            if dup_count > 0:
                issues.append(f"{file_name}: {dup_count} duplicate rows")

            # Check for constant columns
            for col in df.columns:
                if df[col].nunique() == 1:
                    issues.append(f"{file_name}: column '{col}' has only one value")

        return issues

    def generate_summary(
        self,
        file_paths: list[Path],
    ) -> DataSummary:
        """
        Load files and generate a comprehensive data summary.

        Args:
            file_paths: List of paths to data files.

        Returns:
            DataSummary with statistics about all loaded data.
        """
        dataframes = self.load_files(file_paths)

        # Calculate total rows and columns
        total_rows = sum(len(df) for df in dataframes.values())

        # Analyze all columns
        columns = {}
        total_missing = 0
        total_cells = 0

        for file_name, df in dataframes.items():
            for col in df.columns:
                col_info = self.analyze_column(df[col])

                # Use qualified name if column appears in multiple files
                if col in columns:
                    col_key = f"{file_name}:{col}"
                else:
                    col_key = str(col)

                columns[col_key] = col_info

                # Track missing data
                total_missing += df[col].isna().sum()
                total_cells += len(df)

        missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0.0

        # Detect date range and geographic scope
        date_range = self._detect_date_range(dataframes)
        geographic_scope = self._detect_geographic_scope(dataframes)

        # Detect quality issues
        quality_notes = self._detect_data_quality_issues(dataframes)

        return DataSummary(
            files=[str(Path(p).name) for p in file_paths],
            total_rows=total_rows,
            columns=columns,
            missing_data_pct=missing_pct,
            date_range=date_range,
            geographic_scope=geographic_scope,
            data_quality_notes=quality_notes,
        )
