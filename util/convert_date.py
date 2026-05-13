import torch
import math

_MONTHS = {
    b'JAN': 1, b'FEB': 2, b'MAR': 3, b'APR': 4,
    b'MAY': 5, b'JUN': 6, b'JUL': 7, b'AUG': 8,
    b'SEP': 9, b'OCT': 10, b'NOV': 11, b'DEC': 12,
}


_DAYS_BEFORE_MONTH = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]


def _is_leap(year: int) -> bool:
    return (year % 4 == 0) and (year % 100 != 0 or year % 400 == 0)


def _day_of_year(day: int, month: int, year: int) -> int:
    doy = _DAYS_BEFORE_MONTH[month - 1] + day
    if month > 2 and _is_leap(year):
        doy += 1
    return doy

def timestamp_to_vector(timestamp: str) -> torch.Tensor:
    day = (timestamp[0] - 48) * 10 + (timestamp[1] - 48)

    # month: positions 3-5 (3-letter abbrev)
    month = _MONTHS[timestamp[3:6]]

    # year: positions 7-10
    year = ((timestamp[7] - 48) * 1000 +
            (timestamp[8] - 48) * 100 +
            (timestamp[9] - 48) * 10 +
            (timestamp[10] - 48))

    # --- day-of-year ---
    doy = _day_of_year(day, month, year)

    # --- angles ---
    toy_angle = 2.0 * math.pi * (doy / 365.0)  # you can use 365.2422 if you want

    toy_sin = math.sin(toy_angle)
    toy_cos = math.cos(toy_angle)

    return torch.tensor(
        [toy_sin, toy_cos]
    )