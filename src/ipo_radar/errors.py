class SourceError(Exception):
    """A data source failed or changed layout. Callers show it; they never substitute guessed values."""
