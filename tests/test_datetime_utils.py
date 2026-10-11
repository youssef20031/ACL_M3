"""
Tests for datetime utility functions
Includes property-based tests and DST edge case tests.
"""
import unittest
import sys
import os
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.datetime_utils import (
    parse_to_utc,
    is_past,
    to_local_timezone,
    handle_dst_transition,
    get_timezone_offset_hours,
    format_datetime_for_display
)

try:
    from hypothesis import given, strategies as st, assume, settings
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    print("WARNING: hypothesis not installed. Property-based tests will be skipped.")


class TestParseToUTC(unittest.TestCase):
    """Tests for parse_to_utc function."""
    
    def test_parse_dd_mm_yyyy_format(self):
        """Test parsing DD/MM/YYYY format."""
        result = parse_to_utc("15/08/2026", "14:30")
        
        self.assertEqual(result.year, 2026)
        self.assertEqual(result.month, 8)
        self.assertEqual(result.day, 15)
        self.assertEqual(result.hour, 14)
        self.assertEqual(result.minute, 30)
        self.assertEqual(result.tzinfo, timezone.utc)
    
    def test_parse_yyyy_mm_dd_format(self):
        """Test parsing YYYY-MM-DD format."""
        result = parse_to_utc("2026-08-15", "14:30")
        
        self.assertEqual(result.year, 2026)
        self.assertEqual(result.month, 8)
        self.assertEqual(result.day, 15)
        self.assertEqual(result.hour, 14)
        self.assertEqual(result.minute, 30)
        self.assertEqual(result.tzinfo, timezone.utc)
    
    def test_parse_with_seconds(self):
        """Test parsing time with seconds."""
        result = parse_to_utc("15/08/2026", "14:30:45")
        
        self.assertEqual(result.second, 45)
        self.assertEqual(result.tzinfo, timezone.utc)
    
    def test_parse_default_time(self):
        """Test default time is 00:00."""
        result = parse_to_utc("15/08/2026")
        
        self.assertEqual(result.hour, 0)
        self.assertEqual(result.minute, 0)
    
    def test_parse_invalid_date_format(self):
        """Test invalid date format raises ValueError."""
        with self.assertRaises(ValueError):
            parse_to_utc("invalid-date", "15:00")
    
    def test_parse_invalid_time_format(self):
        """Test invalid time format raises ValueError."""
        with self.assertRaises(ValueError):
            parse_to_utc("15/08/2026", "invalid-time")


class TestIsPast(unittest.TestCase):
    """Tests for is_past function."""
    
    def test_past_datetime_returns_true(self):
        """Test that past datetime returns True."""
        past_time = datetime.now(timezone.utc) - timedelta(hours=1)
        self.assertTrue(is_past(past_time))
    
    def test_future_datetime_returns_false(self):
        """Test that future datetime returns False."""
        future_time = datetime.now(timezone.utc) + timedelta(hours=1)
        self.assertFalse(is_past(future_time))
    
    def test_with_custom_reference_time(self):
        """Test is_past with custom reference time."""
        dt = datetime(2026, 8, 15, 14, 0, tzinfo=timezone.utc)
        reference = datetime(2026, 8, 20, 14, 0, tzinfo=timezone.utc)
        
        self.assertTrue(is_past(dt, reference))
    
    def test_naive_datetime_converted_to_utc(self):
        """Test that naive datetime is converted to UTC."""
        # Create naive datetime (no timezone)
        naive_past = datetime.now() - timedelta(hours=1)
        
        # Should still work (converts to UTC)
        result = is_past(naive_past)
        self.assertIsInstance(result, bool)
    
    def test_equal_datetimes(self):
        """Test that equal datetimes returns False (not strictly past)."""
        dt = datetime(2026, 8, 15, 14, 0, tzinfo=timezone.utc)
        self.assertFalse(is_past(dt, dt))


class TestToLocalTimezone(unittest.TestCase):
    """Tests for to_local_timezone function."""
    
    def test_convert_to_london_timezone(self):
        """Test conversion to Europe/London timezone."""
        utc_time = datetime(2026, 8, 15, 14, 0, tzinfo=timezone.utc)
        london_time = to_local_timezone(utc_time, "Europe/London")
        
        # London is UTC+1 during summer (BST)
        self.assertEqual(london_time.hour, 15)
        self.assertEqual(london_time.tzinfo, ZoneInfo("Europe/London"))
    
    def test_convert_to_new_york_timezone(self):
        """Test conversion to America/New_York timezone."""
        utc_time = datetime(2026, 8, 15, 14, 0, tzinfo=timezone.utc)
        ny_time = to_local_timezone(utc_time, "America/New_York")
        
        # New York is UTC-4 during summer (EDT)
        self.assertEqual(ny_time.hour, 10)
        self.assertEqual(ny_time.tzinfo, ZoneInfo("America/New_York"))
    
    def test_invalid_timezone_raises_error(self):
        """Test invalid timezone name raises ValueError."""
        utc_time = datetime(2026, 8, 15, 14, 0, tzinfo=timezone.utc)
        
        with self.assertRaises(ValueError):
            to_local_timezone(utc_time, "Invalid/Timezone")
    
    def test_naive_datetime_converted(self):
        """Test naive datetime is converted to UTC first."""
        naive_time = datetime(2026, 8, 15, 14, 0)
        london_time = to_local_timezone(naive_time, "Europe/London")
        
        # Should convert naive to UTC, then to London (UTC+1 in summer)
        self.assertEqual(london_time.hour, 15)


class TestDSTHandling(unittest.TestCase):
    """Tests for DST (Daylight Saving Time) edge cases."""
    
    def test_spring_forward_gap_us_eastern(self):
        """
        Test spring forward DST transition in US/Eastern.
        
        In 2026, DST starts on March 8 at 2:00 AM.
        Times between 2:00 AM and 3:00 AM don't exist.
        """
        # 2:30 AM doesn't exist on March 8, 2026 in US/Eastern
        non_existent_time = datetime(2026, 3, 8, 2, 30)
        
        # Should handle gracefully (adjust forward to 3:30 AM)
        result = handle_dst_transition(non_existent_time, "America/New_York")
        
        # Result should be timezone-aware
        self.assertIsNotNone(result.tzinfo)
        self.assertEqual(result.tzinfo, ZoneInfo("America/New_York"))
    
    def test_fall_back_overlap_us_eastern(self):
        """
        Test fall back DST transition in US/Eastern.
        
        In 2026, DST ends on November 1 at 2:00 AM.
        Times between 1:00 AM and 2:00 AM occur twice.
        """
        # 1:30 AM occurs twice on November 1, 2026 in US/Eastern
        ambiguous_time = datetime(2026, 11, 1, 1, 30)
        
        # Should handle gracefully (default to first occurrence)
        result = handle_dst_transition(ambiguous_time, "America/New_York")
        
        # Result should be timezone-aware
        self.assertIsNotNone(result.tzinfo)
        self.assertEqual(result.tzinfo, ZoneInfo("America/New_York"))
    
    def test_spring_forward_gap_europe_london(self):
        """
        Test spring forward DST transition in Europe/London.
        
        In 2026, DST starts on March 29 at 1:00 AM.
        Times between 1:00 AM and 2:00 AM don't exist.
        """
        # 1:30 AM doesn't exist on March 29, 2026 in Europe/London
        non_existent_time = datetime(2026, 3, 29, 1, 30)
        
        # Should handle gracefully
        result = handle_dst_transition(non_existent_time, "Europe/London")
        
        self.assertIsNotNone(result.tzinfo)
        self.assertEqual(result.tzinfo, ZoneInfo("Europe/London"))
    
    def test_fall_back_overlap_europe_london(self):
        """
        Test fall back DST transition in Europe/London.
        
        In 2026, DST ends on October 25 at 2:00 AM.
        Times between 1:00 AM and 2:00 AM occur twice.
        """
        # 1:30 AM occurs twice on October 25, 2026 in Europe/London
        ambiguous_time = datetime(2026, 10, 25, 1, 30)
        
        # Should handle gracefully
        result = handle_dst_transition(ambiguous_time, "Europe/London")
        
        self.assertIsNotNone(result.tzinfo)
        self.assertEqual(result.tzinfo, ZoneInfo("Europe/London"))
    
    def test_normal_time_unaffected(self):
        """Test that normal times (not during DST transitions) work correctly."""
        normal_time = datetime(2026, 6, 15, 14, 30)
        
        result = handle_dst_transition(normal_time, "America/New_York")
        
        self.assertEqual(result.hour, 14)
        self.assertEqual(result.minute, 30)


class TestTimezoneOffset(unittest.TestCase):
    """Tests for get_timezone_offset_hours function."""
    
    def test_london_summer_offset(self):
        """Test London timezone offset during summer (BST)."""
        summer = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
        offset = get_timezone_offset_hours("Europe/London", summer)
        
        # London is UTC+1 during summer
        self.assertEqual(offset, 1.0)
    
    def test_london_winter_offset(self):
        """Test London timezone offset during winter (GMT)."""
        winter = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        offset = get_timezone_offset_hours("Europe/London", winter)
        
        # London is UTC+0 during winter
        self.assertEqual(offset, 0.0)
    
    def test_new_york_summer_offset(self):
        """Test New York timezone offset during summer (EDT)."""
        summer = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)
        offset = get_timezone_offset_hours("America/New_York", summer)
        
        # New York is UTC-4 during summer
        self.assertEqual(offset, -4.0)
    
    def test_new_york_winter_offset(self):
        """Test New York timezone offset during winter (EST)."""
        winter = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        offset = get_timezone_offset_hours("America/New_York", winter)
        
        # New York is UTC-5 during winter
        self.assertEqual(offset, -5.0)


class TestFormatDatetimeForDisplay(unittest.TestCase):
    """Tests for format_datetime_for_display function."""
    
    def test_format_with_timezone(self):
        """Test formatting with timezone abbreviation."""
        utc_time = datetime(2026, 8, 15, 14, 30, tzinfo=timezone.utc)
        formatted = format_datetime_for_display(utc_time, "Europe/London", include_timezone=True)
        
        # Should include BST timezone
        self.assertIn("15:30", formatted)
        self.assertIn("Aug", formatted)
        self.assertIn("2026", formatted)
        self.assertIn("BST", formatted)
    
    def test_format_without_timezone(self):
        """Test formatting without timezone abbreviation."""
        utc_time = datetime(2026, 8, 15, 14, 30, tzinfo=timezone.utc)
        formatted = format_datetime_for_display(utc_time, "Europe/London", include_timezone=False)
        
        # Should not include timezone
        self.assertIn("15:30", formatted)
        self.assertNotIn("BST", formatted)


@unittest.skipIf(not HYPOTHESIS_AVAILABLE, "hypothesis not installed")
class TestDatetimeUtilsProperties(unittest.TestCase):
    """Property-based tests for datetime utilities."""
    
    # Property 10: Date Parsing to UTC
    # **Validates: Requirements 5.3, 7.4**
    @given(
        year=st.integers(min_value=2020, max_value=2050),
        month=st.integers(min_value=1, max_value=12),
        day=st.integers(min_value=1, max_value=28),  # Safe for all months
        hour=st.integers(min_value=0, max_value=23),
        minute=st.integers(min_value=0, max_value=59)
    )
    @settings(max_examples=20)
    def test_property_10_date_parsing_to_utc(self, year, month, day, hour, minute):
        """
        Property 10: Date Parsing to UTC
        
        For any valid date and time string, parse_to_utc SHALL return a 
        timezone-aware datetime in UTC.
        
        **Validates: Requirements 5.3, 7.4**
        """
        # Format date strings
        date_str_ddmmyyyy = f"{day:02d}/{month:02d}/{year}"
        date_str_yyyymmdd = f"{year}-{month:02d}-{day:02d}"
        time_str = f"{hour:02d}:{minute:02d}"
        
        # Test DD/MM/YYYY format
        result1 = parse_to_utc(date_str_ddmmyyyy, time_str)
        
        # Should be timezone-aware in UTC
        self.assertIsNotNone(result1.tzinfo, "Result should be timezone-aware")
        self.assertEqual(result1.tzinfo, timezone.utc, "Result should be in UTC")
        
        # Should match input values
        self.assertEqual(result1.year, year)
        self.assertEqual(result1.month, month)
        self.assertEqual(result1.day, day)
        self.assertEqual(result1.hour, hour)
        self.assertEqual(result1.minute, minute)
        
        # Test YYYY-MM-DD format
        result2 = parse_to_utc(date_str_yyyymmdd, time_str)
        
        # Should be timezone-aware in UTC
        self.assertIsNotNone(result2.tzinfo, "Result should be timezone-aware")
        self.assertEqual(result2.tzinfo, timezone.utc, "Result should be in UTC")
        
        # Both formats should produce the same result
        self.assertEqual(result1, result2, "Both date formats should parse to same datetime")
    
    # Property 15: UTC Storage Invariant
    # **Validates: Requirements 7.1, 7.2**
    @given(
        year=st.integers(min_value=2020, max_value=2050),
        month=st.integers(min_value=1, max_value=12),
        day=st.integers(min_value=1, max_value=28),
        hour=st.integers(min_value=0, max_value=23),
        minute=st.integers(min_value=0, max_value=59)
    )
    @settings(max_examples=20)
    def test_property_15_utc_storage_invariant(self, year, month, day, hour, minute):
        """
        Property 15: UTC Storage Invariant
        
        For any datetime stored in the system, the tzinfo attribute SHALL be 
        timezone.utc.
        
        **Validates: Requirements 7.1, 7.2**
        """
        # Create datetime
        dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
        
        # Verify UTC invariant
        self.assertIsNotNone(dt.tzinfo, "Datetime should be timezone-aware")
        self.assertEqual(dt.tzinfo, timezone.utc, "Datetime should be in UTC")
        
        # Parse from string
        date_str = f"{day:02d}/{month:02d}/{year}"
        time_str = f"{hour:02d}:{minute:02d}"
        parsed_dt = parse_to_utc(date_str, time_str)
        
        # Parsed datetime should also be in UTC
        self.assertEqual(parsed_dt.tzinfo, timezone.utc, "Parsed datetime should be in UTC")
    
    # Property 16: Timezone Conversion Preserves Time
    # **Validates: Requirements 7.3**
    @given(
        year=st.integers(min_value=2020, max_value=2050),
        month=st.integers(min_value=1, max_value=12),
        day=st.integers(min_value=1, max_value=28),
        hour=st.integers(min_value=0, max_value=23),
        minute=st.integers(min_value=0, max_value=59),
        tz_name=st.sampled_from(["Europe/London", "America/New_York", "Asia/Tokyo", "Australia/Sydney"])
    )
    @settings(max_examples=20)
    def test_property_16_timezone_conversion_preserves_time(self, year, month, day, hour, minute, tz_name):
        """
        Property 16: Timezone Conversion Preserves Time
        
        For any UTC datetime and target timezone, converting to local timezone 
        and back to UTC SHALL preserve the original UTC time.
        
        **Validates: Requirements 7.3**
        """
        # Create UTC datetime
        utc_dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
        
        # Convert to local timezone
        local_dt = to_local_timezone(utc_dt, tz_name)
        
        # Convert back to UTC
        back_to_utc = local_dt.astimezone(timezone.utc)
        
        # Should preserve the original UTC time
        self.assertEqual(back_to_utc, utc_dt, 
                        f"Converting to {tz_name} and back should preserve UTC time")
        
        # The absolute time should be the same (timestamps equal)
        self.assertEqual(back_to_utc.timestamp(), utc_dt.timestamp(),
                        "Timestamps should be equal after round-trip conversion")


if __name__ == '__main__':
    unittest.main()
