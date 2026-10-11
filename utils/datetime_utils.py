"""
Date/Time Utility Functions
Provides consistent timezone handling across the application.
"""
from typing import Optional
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import logging

logger = logging.getLogger(__name__)


def parse_to_utc(date_str: str, time_str: str = "00:00") -> datetime:
    """
    Parse date and time strings into a UTC datetime object.
    
    Args:
        date_str: Date string in DD/MM/YYYY or YYYY-MM-DD format
        time_str: Time string in HH:MM or HH:MM:SS format (default: "00:00")
    
    Returns:
        datetime object in UTC timezone
    
    Raises:
        ValueError: If date/time format is invalid
    
    Examples:
        >>> parse_to_utc("15/08/2026", "15:00")
        datetime.datetime(2026, 8, 15, 15, 0, tzinfo=datetime.timezone.utc)
        
        >>> parse_to_utc("2026-08-15", "17:30")
        datetime.datetime(2026, 8, 15, 17, 30, tzinfo=datetime.timezone.utc)
    """
    # Try DD/MM/YYYY format first
    try:
        dt = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M")
    except ValueError:
        try:
            # Try YYYY-MM-DD format
            dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        except ValueError:
            try:
                # Try with seconds
                dt = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M:%S")
            except ValueError:
                try:
                    # Try YYYY-MM-DD with seconds
                    dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
                except ValueError as e:
                    raise ValueError(f"Invalid date/time format: {date_str} {time_str}") from e
    
    # Convert to UTC
    dt_utc = dt.replace(tzinfo=timezone.utc)
    
    logger.debug(f"Parsed '{date_str} {time_str}' to UTC: {dt_utc}")
    return dt_utc


def is_past(datetime_utc: datetime, reference_time: Optional[datetime] = None) -> bool:
    """
    Check if a datetime is in the past.
    
    Args:
        datetime_utc: Datetime to check (should be UTC)
        reference_time: Reference time to compare against (defaults to now in UTC)
    
    Returns:
        True if datetime_utc is in the past, False otherwise
    
    Examples:
        >>> from datetime import datetime, timezone, timedelta
        >>> past_time = datetime.now(timezone.utc) - timedelta(hours=1)
        >>> is_past(past_time)
        True
        
        >>> future_time = datetime.now(timezone.utc) + timedelta(hours=1)
        >>> is_past(future_time)
        False
    """
    if reference_time is None:
        reference_time = datetime.now(timezone.utc)
    
    # Ensure both datetimes are timezone-aware in UTC
    if datetime_utc.tzinfo is None:
        datetime_utc = datetime_utc.replace(tzinfo=timezone.utc)
        logger.warning(f"Naive datetime converted to UTC: {datetime_utc}")
    
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)
    
    return datetime_utc < reference_time


def to_local_timezone(datetime_utc: datetime, tz_name: str) -> datetime:
    """
    Convert UTC datetime to local timezone.
    
    Args:
        datetime_utc: UTC datetime to convert
        tz_name: Timezone name (e.g., "Europe/London", "America/New_York")
    
    Returns:
        datetime object in the specified timezone
    
    Raises:
        ValueError: If timezone name is invalid
    
    Examples:
        >>> from datetime import datetime, timezone
        >>> utc_time = datetime(2026, 8, 15, 14, 0, tzinfo=timezone.utc)
        >>> to_local_timezone(utc_time, "Europe/London")
        datetime.datetime(2026, 8, 15, 15, 0, tzinfo=zoneinfo.ZoneInfo(key='Europe/London'))
        
        >>> to_local_timezone(utc_time, "America/New_York")
        datetime.datetime(2026, 8, 15, 10, 0, tzinfo=zoneinfo.ZoneInfo(key='America/New_York'))
    """
    # Ensure datetime is timezone-aware in UTC
    if datetime_utc.tzinfo is None:
        datetime_utc = datetime_utc.replace(tzinfo=timezone.utc)
        logger.warning(f"Naive datetime converted to UTC: {datetime_utc}")
    
    try:
        local_tz = ZoneInfo(tz_name)
    except Exception as e:
        raise ValueError(f"Invalid timezone name: {tz_name}") from e
    
    # Convert to local timezone
    local_time = datetime_utc.astimezone(local_tz)
    
    logger.debug(f"Converted {datetime_utc} to {tz_name}: {local_time}")
    return local_time


def handle_dst_transition(dt: datetime, tz_name: str) -> datetime:
    """
    Handle daylight saving time transitions safely.
    
    This function ensures that datetimes during DST transitions are handled correctly:
    - During spring forward (gap): non-existent times are adjusted forward
    - During fall back (overlap): ambiguous times default to standard time
    
    Args:
        dt: Naive or timezone-aware datetime
        tz_name: Timezone name (e.g., "Europe/London", "America/New_York")
    
    Returns:
        Timezone-aware datetime with DST handled correctly
    
    Examples:
        >>> from datetime import datetime
        >>> # Spring forward: 2am becomes 3am in US/Eastern
        >>> spring_gap = datetime(2026, 3, 8, 2, 30)
        >>> handle_dst_transition(spring_gap, "America/New_York")
        datetime.datetime(2026, 3, 8, 3, 30, tzinfo=zoneinfo.ZoneInfo(key='America/New_York'))
    """
    try:
        local_tz = ZoneInfo(tz_name)
    except Exception as e:
        raise ValueError(f"Invalid timezone name: {tz_name}") from e
    
    # If already timezone-aware, convert to target timezone
    if dt.tzinfo is not None:
        return dt.astimezone(local_tz)
    
    # For naive datetimes, use replace to attach timezone
    # This will raise an exception for non-existent times (spring forward gap)
    # or use fold parameter to handle ambiguous times (fall back overlap)
    try:
        # Try to localize with fold=0 (first occurrence during fall back)
        localized = dt.replace(tzinfo=local_tz)
        return localized
    except Exception:
        # If time doesn't exist (spring forward gap), adjust forward
        logger.warning(f"Non-existent time during DST transition: {dt} in {tz_name}")
        # Convert to UTC first, then back to local
        utc_dt = dt.replace(tzinfo=timezone.utc)
        return utc_dt.astimezone(local_tz)


def get_timezone_offset_hours(tz_name: str, dt: Optional[datetime] = None) -> float:
    """
    Get the UTC offset in hours for a timezone at a specific datetime.
    
    Args:
        tz_name: Timezone name
        dt: Datetime to check offset for (defaults to now)
    
    Returns:
        UTC offset in hours (e.g., -5.0 for EST, 1.0 for BST)
    
    Examples:
        >>> from datetime import datetime, timezone
        >>> # London is UTC+1 during summer (BST)
        >>> summer = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
        >>> get_timezone_offset_hours("Europe/London", summer)
        1.0
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    
    try:
        local_tz = ZoneInfo(tz_name)
    except Exception as e:
        raise ValueError(f"Invalid timezone name: {tz_name}") from e
    
    # Convert to target timezone and get offset
    local_time = dt.astimezone(local_tz)
    offset_seconds = local_time.utcoffset().total_seconds()
    offset_hours = offset_seconds / 3600
    
    return offset_hours


def format_datetime_for_display(dt: datetime, tz_name: str = "Europe/London", 
                                 include_timezone: bool = True) -> str:
    """
    Format datetime for user-friendly display.
    
    Args:
        dt: Datetime to format (should be UTC)
        tz_name: Timezone to display in (default: "Europe/London")
        include_timezone: Whether to include timezone abbreviation
    
    Returns:
        Formatted datetime string
    
    Examples:
        >>> from datetime import datetime, timezone
        >>> utc_time = datetime(2026, 8, 15, 14, 30, tzinfo=timezone.utc)
        >>> format_datetime_for_display(utc_time, "Europe/London")
        'Sat 15 Aug 2026, 15:30 BST'
    """
    # Convert to local timezone
    local_time = to_local_timezone(dt, tz_name)
    
    # Format: "Sat 15 Aug 2026, 15:30 BST"
    if include_timezone:
        return local_time.strftime("%a %d %b %Y, %H:%M %Z")
    else:
        return local_time.strftime("%a %d %b %Y, %H:%M")
