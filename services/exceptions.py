"""
Custom exception classes for Fixture Service
Provides descriptive error handling for fixture-related operations.
"""


class FixtureServiceError(Exception):
    """Base exception for all fixture service errors."""
    
    def __init__(self, message: str, fixture_id: str = None, details: dict = None):
        """
        Initialize fixture service error.
        
        Args:
            message: Error message
            fixture_id: Optional fixture identifier
            details: Optional dictionary with additional error details
        """
        self.message = message
        self.fixture_id = fixture_id
        self.details = details or {}
        
        # Build full error message
        full_message = message
        if fixture_id:
            full_message = f"{message} (fixture_id: {fixture_id})"
        if details:
            full_message = f"{full_message} - Details: {details}"
        
        super().__init__(full_message)


class FixtureNotFoundError(FixtureServiceError):
    """Exception raised when a fixture is not found in the database."""
    
    def __init__(self, fixture_id: str, message: str = None):
        """
        Initialize fixture not found error.
        
        Args:
            fixture_id: Fixture identifier that was not found
            message: Optional custom error message
        """
        if message is None:
            message = f"Fixture not found: {fixture_id}"
        super().__init__(message, fixture_id=fixture_id)


class InvalidFixtureDataError(FixtureServiceError):
    """Exception raised when fixture data is invalid or malformed."""
    
    def __init__(self, message: str, fixture_id: str = None, details: dict = None):
        """
        Initialize invalid fixture data error.
        
        Args:
            message: Description of the validation error
            fixture_id: Optional fixture identifier
            details: Optional dictionary with validation details
        """
        super().__init__(message, fixture_id=fixture_id, details=details)


class DuplicateFixtureError(FixtureServiceError):
    """Exception raised when attempting to create a duplicate fixture."""
    
    def __init__(self, fixture_id: str, message: str = None):
        """
        Initialize duplicate fixture error.
        
        Args:
            fixture_id: Fixture identifier that already exists
            message: Optional custom error message
        """
        if message is None:
            message = f"Fixture already exists: {fixture_id}"
        super().__init__(message, fixture_id=fixture_id)


class FixtureValidationError(InvalidFixtureDataError):
    """Exception raised when fixture fails validation checks."""
    
    def __init__(self, validation_errors: list, fixture_id: str = None):
        """
        Initialize fixture validation error.
        
        Args:
            validation_errors: List of validation error messages
            fixture_id: Optional fixture identifier
        """
        message = f"Fixture validation failed: {', '.join(validation_errors)}"
        details = {"validation_errors": validation_errors}
        super().__init__(message, fixture_id=fixture_id, details=details)


class TeamNotFoundError(FixtureServiceError):
    """Exception raised when a team referenced in fixture data is not found."""
    
    def __init__(self, team_name: str, fixture_id: str = None):
        """
        Initialize team not found error.
        
        Args:
            team_name: Team name that was not found
            fixture_id: Optional fixture identifier
        """
        message = f"Team not found in database: {team_name}"
        details = {"team_name": team_name}
        super().__init__(message, fixture_id=fixture_id, details=details)


class InvalidDateFormatError(InvalidFixtureDataError):
    """Exception raised when date/time format is invalid."""
    
    def __init__(self, date_str: str, expected_format: str = None):
        """
        Initialize invalid date format error.
        
        Args:
            date_str: Date string that failed to parse
            expected_format: Optional expected date format description
        """
        message = f"Invalid date format: '{date_str}'"
        if expected_format:
            message = f"{message}. Expected format: {expected_format}"
        
        details = {"date_str": date_str, "expected_format": expected_format}
        super().__init__(message, details=details)


class CSVLoadError(FixtureServiceError):
    """Exception raised when CSV file loading fails."""
    
    def __init__(self, csv_path: str, message: str, line_number: int = None):
        """
        Initialize CSV load error.
        
        Args:
            csv_path: Path to CSV file
            message: Error description
            line_number: Optional line number where error occurred
        """
        full_message = f"CSV load error in '{csv_path}': {message}"
        if line_number is not None:
            full_message = f"{full_message} (line {line_number})"
        
        details = {"csv_path": csv_path, "line_number": line_number}
        super().__init__(full_message, details=details)
