"""
create_trading_holidays.py
──────────────────────────
Creates Trading_Holidays.xlsx with one sheet per exchange,
covering the data range 2023-01-23 to 2026-02-20.

Sheets:
  NIFTY_Holidays   → NSE India    (NIFTY, INVIXN)
  DJ_Holidays      → NYSE         (INDU / Dow Jones)
  SP_Holidays      → NYSE/CBOE    (SPX, S&P 500)
  DAX_Holidays     → XETRA        (DAX, VXEFA, VXEEM, V2X)
  UKX_Holidays     → LSE          (UKX / FTSE 100)
  HSI_Holidays     → HKEX         (HSI, VHSI)
  SHCOMP_Holidays  → SSE          (SHCOMP / Shanghai)
  TWSE_Holidays    → TWSE         (Taiwan)
  NKY_Holidays     → TSE          (NKY / Nikkei)
  STI_Holidays     → SGX          (STI / Singapore)
"""

import pandas as pd
import os

input_fp = r"C:\Users\Manpreet\OneDrive - City St George's, University of London\Documents\Term 2\SMM748 Machine Learning Quantitative Finance\First Group Coursework ML\MS_Python_CW\input_data"

# ──────────────────────────────────────────────
# NSE India — NIFTY_Holidays
# ──────────────────────────────────────────────
nifty_holidays = [
    # 2023
    ("2023-01-26", "Thu", "Republic Day"),
    ("2023-03-07", "Tue", "Holi"),
    ("2023-03-30", "Thu", "Ram Navami"),
    ("2023-04-04", "Tue", "Mahavir Jayanti"),
    ("2023-04-07", "Fri", "Good Friday"),
    ("2023-04-14", "Fri", "Dr. Ambedkar Jayanti"),
    ("2023-04-22", "Sat", "Id-Ul-Fitr"),  # weekend
    ("2023-05-01", "Mon", "Maharashtra Day"),
    ("2023-06-29", "Thu", "Bakri Id"),
    ("2023-06-29", "Thu", "Bakri Id"),
    ("2023-08-15", "Tue", "Independence Day"),
    ("2023-09-19", "Tue", "Milad-Un-Nabi"),
    ("2023-10-02", "Mon", "Gandhi Jayanti"),
    ("2023-10-24", "Tue", "Dussehra"),
    ("2023-11-14", "Tue", "Diwali (Laxmi Pujan)"),
    ("2023-11-27", "Mon", "Guru Nanak Jayanti"),
    ("2023-12-25", "Mon", "Christmas"),
    # 2024
    ("2024-01-22", "Mon", "Special Day"),
    ("2024-01-26", "Fri", "Republic Day"),
    ("2024-03-08", "Fri", "Mahashivratri"),
    ("2024-03-25", "Mon", "Holi"),
    ("2024-03-29", "Fri", "Good Friday"),
    ("2024-04-11", "Thu", "Id-Ul-Fitr"),
    ("2024-04-17", "Wed", "Shri Ram Navami"),
    ("2024-05-01", "Wed", "Maharashtra Day"),
    ("2024-05-20", "Mon", "Parliamentary Elections"),
    ("2024-06-17", "Mon", "Bakri Id"),
    ("2024-07-17", "Wed", "Moharram"),
    ("2024-08-15", "Thu", "Independence Day"),
    ("2024-10-02", "Wed", "Gandhi Jayanti"),
    ("2024-11-01", "Fri", "Diwali (Laxmi Pujan)"),
    ("2024-11-15", "Fri", "Gurunanak Jayanti"),
    ("2024-12-25", "Wed", "Christmas"),
    # 2025
    ("2025-02-26", "Wed", "Maha Shivaratri"),
    ("2025-03-14", "Fri", "Holi"),
    ("2025-03-31", "Mon", "Id-Ul-Fitr"),
    ("2025-04-10", "Thu", "Ram Navami"),
    ("2025-04-14", "Mon", "Ambedkar Jayanti"),
    ("2025-04-18", "Fri", "Good Friday"),
    ("2025-05-01", "Thu", "Maharashtra Day"),
    ("2025-06-06", "Fri", "Id-Ul-Zuha"),
    ("2025-08-15", "Fri", "Independence Day"),
    ("2025-10-02", "Thu", "Gandhi Jayanti"),
    ("2025-10-21", "Tue", "Diwali (Muhurat Trading)"),
    ("2025-10-22", "Wed", "Diwali Balipratipada"),
    ("2025-11-03", "Mon", "Guru Nanak Jayanti"),
    ("2025-12-25", "Thu", "Christmas"),
    # 2026
    ("2026-01-26", "Mon", "Republic Day"),
]

# ──────────────────────────────────────────────
# NYSE — DJ_Holidays (INDU) & SP_Holidays (SPX)
#   (NYSE and CBOE share the same holiday schedule)
# ──────────────────────────────────────────────
usa_holidays = [
    # 2023
    ("2023-01-16", "Mon", "Martin Luther King Jr. Day"),
    ("2023-02-20", "Mon", "Presidents' Day"),
    ("2023-04-07", "Fri", "Good Friday"),
    ("2023-05-29", "Mon", "Memorial Day"),
    ("2023-06-19", "Mon", "Juneteenth"),
    ("2023-07-04", "Tue", "Independence Day"),
    ("2023-09-04", "Mon", "Labor Day"),
    ("2023-11-23", "Thu", "Thanksgiving Day"),
    ("2023-12-25", "Mon", "Christmas Day"),
    # 2024
    ("2024-01-01", "Mon", "New Year's Day"),
    ("2024-01-15", "Mon", "Martin Luther King Jr. Day"),
    ("2024-02-19", "Mon", "Presidents' Day"),
    ("2024-03-29", "Fri", "Good Friday"),
    ("2024-05-27", "Mon", "Memorial Day"),
    ("2024-06-19", "Wed", "Juneteenth"),
    ("2024-07-04", "Thu", "Independence Day"),
    ("2024-09-02", "Mon", "Labor Day"),
    ("2024-11-28", "Thu", "Thanksgiving Day"),
    ("2024-12-25", "Wed", "Christmas Day"),
    # 2025
    ("2025-01-01", "Wed", "New Year's Day"),
    ("2025-01-09", "Thu", "National Day of Mourning (Jimmy Carter)"),
    ("2025-01-20", "Mon", "Martin Luther King Jr. Day"),
    ("2025-02-17", "Mon", "Presidents' Day"),
    ("2025-04-18", "Fri", "Good Friday"),
    ("2025-05-26", "Mon", "Memorial Day"),
    ("2025-06-19", "Thu", "Juneteenth"),
    ("2025-07-04", "Fri", "Independence Day"),
    ("2025-09-01", "Mon", "Labor Day"),
    ("2025-11-27", "Thu", "Thanksgiving Day"),
    ("2025-12-25", "Thu", "Christmas Day"),
    # 2026
    ("2026-01-01", "Thu", "New Year's Day"),
    ("2026-01-19", "Mon", "Martin Luther King Jr. Day"),
    ("2026-02-16", "Mon", "Presidents' Day"),
]

# ──────────────────────────────────────────────
# XETRA — DAX_Holidays (also VXEFA, VXEEM, V2X)
# ──────────────────────────────────────────────
dax_holidays = [
    # 2023
    ("2023-04-07", "Fri", "Good Friday"),
    ("2023-04-10", "Mon", "Easter Monday"),
    ("2023-05-01", "Mon", "Labour Day"),
    ("2023-12-25", "Mon", "Christmas Day"),
    ("2023-12-26", "Tue", "St. Stephen's Day"),
    # 2024
    ("2024-03-29", "Fri", "Good Friday"),
    ("2024-04-01", "Mon", "Easter Monday"),
    ("2024-05-01", "Wed", "Labour Day"),
    ("2024-12-24", "Tue", "Christmas Eve"),
    ("2024-12-25", "Wed", "Christmas Day"),
    ("2024-12-26", "Thu", "St. Stephen's Day"),
    ("2024-12-31", "Tue", "New Year's Eve"),
    # 2025
    ("2025-01-01", "Wed", "New Year's Day"),
    ("2025-04-18", "Fri", "Good Friday"),
    ("2025-04-21", "Mon", "Easter Monday"),
    ("2025-05-01", "Thu", "Labour Day"),
    ("2025-12-24", "Wed", "Christmas Eve"),
    ("2025-12-25", "Thu", "Christmas Day"),
    ("2025-12-26", "Fri", "St. Stephen's Day"),
    ("2025-12-31", "Wed", "New Year's Eve"),
    # 2026
    ("2026-01-01", "Thu", "New Year's Day"),
]

# ──────────────────────────────────────────────
# LSE — UKX_Holidays (FTSE 100)
# ──────────────────────────────────────────────
ukx_holidays = [
    # 2023
    ("2023-04-07", "Fri", "Good Friday"),
    ("2023-04-10", "Mon", "Easter Monday"),
    ("2023-05-01", "Mon", "Early May Bank Holiday"),
    ("2023-05-08", "Mon", "Coronation Bank Holiday"),
    ("2023-05-29", "Mon", "Spring Bank Holiday"),
    ("2023-08-28", "Mon", "Summer Bank Holiday"),
    ("2023-12-25", "Mon", "Christmas Day"),
    ("2023-12-26", "Tue", "Boxing Day"),
    # 2024
    ("2024-01-01", "Mon", "New Year's Day"),
    ("2024-03-29", "Fri", "Good Friday"),
    ("2024-04-01", "Mon", "Easter Monday"),
    ("2024-05-06", "Mon", "Early May Bank Holiday"),
    ("2024-05-27", "Mon", "Spring Bank Holiday"),
    ("2024-08-26", "Mon", "Summer Bank Holiday"),
    ("2024-12-25", "Wed", "Christmas Day"),
    ("2024-12-26", "Thu", "Boxing Day"),
    # 2025
    ("2025-01-01", "Wed", "New Year's Day"),
    ("2025-04-18", "Fri", "Good Friday"),
    ("2025-04-21", "Mon", "Easter Monday"),
    ("2025-05-05", "Mon", "Early May Bank Holiday"),
    ("2025-05-26", "Mon", "Spring Bank Holiday"),
    ("2025-08-25", "Mon", "Summer Bank Holiday"),
    ("2025-12-25", "Thu", "Christmas Day"),
    ("2025-12-26", "Fri", "Boxing Day"),
    # 2026
    ("2026-01-01", "Thu", "New Year's Day"),
]

# ──────────────────────────────────────────────
# HKEX — HSI_Holidays (HSI, VHSI)
# ──────────────────────────────────────────────
hsi_holidays = [
    # 2023
    ("2023-01-23", "Mon", "Lunar New Year Day 2"),
    ("2023-01-24", "Tue", "Lunar New Year Day 3"),
    ("2023-01-25", "Wed", "Lunar New Year Day 4"),
    ("2023-04-05", "Wed", "Ching Ming Festival"),
    ("2023-04-07", "Fri", "Good Friday"),
    ("2023-04-10", "Mon", "Easter Monday"),
    ("2023-05-01", "Mon", "Labour Day"),
    ("2023-05-26", "Fri", "Buddha's Birthday"),
    ("2023-06-22", "Thu", "Tuen Ng Festival"),
    ("2023-07-01", "Sat", "HKSAR Establishment Day"),  # weekend
    ("2023-09-29", "Fri", "Day after Mid-Autumn"),
    ("2023-10-02", "Mon", "National Day (observed)"),
    ("2023-10-23", "Mon", "Chung Yeung Festival"),
    ("2023-12-25", "Mon", "Christmas Day"),
    ("2023-12-26", "Tue", "Boxing Day"),
    # 2024
    ("2024-02-12", "Mon", "Lunar New Year Day 3"),
    ("2024-02-13", "Tue", "Lunar New Year Day 4 (extra)"),
    ("2024-03-29", "Fri", "Good Friday"),
    ("2024-04-01", "Mon", "Easter Monday"),
    ("2024-04-04", "Thu", "Ching Ming Festival"),
    ("2024-05-01", "Wed", "Labour Day"),
    ("2024-05-15", "Wed", "Buddha's Birthday"),
    ("2024-06-10", "Mon", "Tuen Ng Festival"),
    ("2024-07-01", "Mon", "HKSAR Establishment Day"),
    ("2024-09-18", "Wed", "Day after Mid-Autumn"),
    ("2024-10-01", "Tue", "National Day"),
    ("2024-10-11", "Fri", "Chung Yeung Festival"),
    ("2024-12-25", "Wed", "Christmas Day"),
    ("2024-12-26", "Thu", "Boxing Day"),
    # 2025
    ("2025-01-01", "Wed", "New Year's Day"),
    ("2025-01-29", "Wed", "Lunar New Year Day 1"),
    ("2025-01-30", "Thu", "Lunar New Year Day 2"),
    ("2025-01-31", "Fri", "Lunar New Year Day 3"),
    ("2025-04-04", "Fri", "Ching Ming Festival"),
    ("2025-04-18", "Fri", "Good Friday"),
    ("2025-04-21", "Mon", "Easter Monday"),
    ("2025-05-01", "Thu", "Labour Day"),
    ("2025-05-05", "Mon", "Buddha's Birthday"),
    ("2025-07-01", "Tue", "HKSAR Establishment Day"),
    ("2025-10-01", "Wed", "National Day"),
    ("2025-10-07", "Tue", "Day after Mid-Autumn"),
    ("2025-10-29", "Wed", "Chung Yeung Festival"),
    ("2025-12-25", "Thu", "Christmas Day"),
    ("2025-12-26", "Fri", "Boxing Day"),
    # 2026
    ("2026-01-01", "Thu", "New Year's Day"),
    ("2026-02-17", "Tue", "Lunar New Year Day 1"),
    ("2026-02-18", "Wed", "Lunar New Year Day 2"),
    ("2026-02-19", "Thu", "Lunar New Year Day 3"),
]

# ──────────────────────────────────────────────
# SSE — SHCOMP_Holidays (Shanghai Composite)
# ──────────────────────────────────────────────
shcomp_holidays = [
    # 2023
    ("2023-01-23", "Mon", "Lunar New Year"),
    ("2023-01-24", "Tue", "Lunar New Year"),
    ("2023-01-25", "Wed", "Lunar New Year"),
    ("2023-01-26", "Thu", "Lunar New Year"),
    ("2023-01-27", "Fri", "Lunar New Year"),
    ("2023-04-05", "Wed", "Qingming Festival"),
    ("2023-05-01", "Mon", "Labour Day"),
    ("2023-05-02", "Tue", "Labour Day"),
    ("2023-05-03", "Wed", "Labour Day"),
    ("2023-06-22", "Thu", "Dragon Boat Festival"),
    ("2023-06-23", "Fri", "Dragon Boat Festival"),
    ("2023-09-29", "Fri", "Mid-Autumn Festival"),
    ("2023-10-02", "Mon", "National Day"),
    ("2023-10-03", "Tue", "National Day"),
    ("2023-10-04", "Wed", "National Day"),
    ("2023-10-05", "Thu", "National Day"),
    ("2023-10-06", "Fri", "National Day"),
    # 2024
    ("2024-02-09", "Fri", "Lunar New Year"),
    ("2024-02-12", "Mon", "Lunar New Year"),
    ("2024-02-13", "Tue", "Lunar New Year"),
    ("2024-02-14", "Wed", "Lunar New Year"),
    ("2024-02-15", "Thu", "Lunar New Year"),
    ("2024-02-16", "Fri", "Lunar New Year"),
    ("2024-04-04", "Thu", "Qingming Festival"),
    ("2024-04-05", "Fri", "Qingming Festival"),
    ("2024-05-01", "Wed", "Labour Day"),
    ("2024-05-02", "Thu", "Labour Day"),
    ("2024-05-03", "Fri", "Labour Day"),
    ("2024-06-10", "Mon", "Dragon Boat Festival"),
    ("2024-09-16", "Mon", "Mid-Autumn Festival"),
    ("2024-09-17", "Tue", "Mid-Autumn Festival"),
    ("2024-10-01", "Tue", "National Day"),
    ("2024-10-02", "Wed", "National Day"),
    ("2024-10-03", "Thu", "National Day"),
    ("2024-10-04", "Fri", "National Day"),
    ("2024-10-07", "Mon", "National Day"),
    # 2025
    ("2025-01-01", "Wed", "New Year's Day"),
    ("2025-01-28", "Tue", "Lunar New Year"),
    ("2025-01-29", "Wed", "Lunar New Year"),
    ("2025-01-30", "Thu", "Lunar New Year"),
    ("2025-01-31", "Fri", "Lunar New Year"),
    ("2025-02-03", "Mon", "Lunar New Year"),
    ("2025-02-04", "Tue", "Lunar New Year"),
    ("2025-04-04", "Fri", "Qingming Festival"),
    ("2025-05-01", "Thu", "Labour Day"),
    ("2025-05-02", "Fri", "Labour Day"),
    ("2025-05-05", "Mon", "Labour Day (bridge)"),
    ("2025-05-31", "Sat", "Dragon Boat Festival"),  # weekend
    ("2025-06-02", "Mon", "Dragon Boat Festival (bridge)"),
    ("2025-10-01", "Wed", "National Day"),
    ("2025-10-02", "Thu", "National Day"),
    ("2025-10-03", "Fri", "National Day"),
    ("2025-10-06", "Mon", "National Day"),
    ("2025-10-07", "Tue", "National Day"),
    # 2026
    ("2026-01-01", "Thu", "New Year's Day"),
    ("2026-02-17", "Tue", "Lunar New Year"),
    ("2026-02-18", "Wed", "Lunar New Year"),
    ("2026-02-19", "Thu", "Lunar New Year"),
    ("2026-02-20", "Fri", "Lunar New Year"),
]

# ──────────────────────────────────────────────
# TWSE — TWSE_Holidays (Taiwan Stock Exchange)
# ──────────────────────────────────────────────
twse_holidays = [
    # 2023
    ("2023-01-20", "Fri", "Lunar New Year"),
    ("2023-01-23", "Mon", "Lunar New Year"),
    ("2023-01-24", "Tue", "Lunar New Year"),
    ("2023-01-25", "Wed", "Lunar New Year"),
    ("2023-01-26", "Thu", "Lunar New Year"),
    ("2023-01-27", "Fri", "Lunar New Year (adjusted)"),
    ("2023-02-27", "Mon", "Peace Memorial Day (observed)"),
    ("2023-02-28", "Tue", "Peace Memorial Day"),
    ("2023-04-03", "Mon", "Children's Day (bridge)"),
    ("2023-04-04", "Tue", "Children's Day"),
    ("2023-04-05", "Wed", "Qingming Festival"),
    ("2023-05-01", "Mon", "Labour Day"),
    ("2023-06-22", "Thu", "Dragon Boat Festival"),
    ("2023-06-23", "Fri", "Dragon Boat Festival (bridge)"),
    ("2023-09-29", "Fri", "Mid-Autumn Festival"),
    ("2023-10-09", "Mon", "National Day (bridge)"),
    ("2023-10-10", "Tue", "National Day"),
    # 2024
    ("2024-02-08", "Thu", "Lunar New Year"),
    ("2024-02-09", "Fri", "Lunar New Year"),
    ("2024-02-12", "Mon", "Lunar New Year"),
    ("2024-02-13", "Tue", "Lunar New Year"),
    ("2024-02-14", "Wed", "Lunar New Year"),
    ("2024-02-28", "Wed", "Peace Memorial Day"),
    ("2024-04-04", "Thu", "Children's Day / Qingming"),
    ("2024-04-05", "Fri", "Qingming Festival (bridge)"),
    ("2024-05-01", "Wed", "Labour Day"),
    ("2024-06-10", "Mon", "Dragon Boat Festival"),
    ("2024-09-17", "Tue", "Mid-Autumn Festival"),
    ("2024-10-10", "Thu", "National Day"),
    # 2025
    ("2025-01-01", "Wed", "New Year's Day"),
    ("2025-01-27", "Mon", "Lunar New Year"),
    ("2025-01-28", "Tue", "Lunar New Year"),
    ("2025-01-29", "Wed", "Lunar New Year"),
    ("2025-01-30", "Thu", "Lunar New Year"),
    ("2025-01-31", "Fri", "Lunar New Year"),
    ("2025-02-28", "Fri", "Peace Memorial Day"),
    ("2025-04-03", "Thu", "Children's Day"),
    ("2025-04-04", "Fri", "Qingming Festival"),
    ("2025-05-01", "Thu", "Labour Day"),
    ("2025-05-02", "Fri", "Labour Day (bridge)"),
    ("2025-05-30", "Fri", "Dragon Boat Festival (bridge)"),
    ("2025-10-06", "Mon", "Mid-Autumn Festival (observed)"),
    ("2025-10-10", "Fri", "National Day"),
    # 2026
    ("2026-01-01", "Thu", "New Year's Day"),
    ("2026-02-16", "Mon", "Lunar New Year"),
    ("2026-02-17", "Tue", "Lunar New Year"),
    ("2026-02-18", "Wed", "Lunar New Year"),
    ("2026-02-19", "Thu", "Lunar New Year"),
    ("2026-02-20", "Fri", "Lunar New Year"),
]

# ──────────────────────────────────────────────
# TSE — NKY_Holidays (Nikkei / Japan)
# ──────────────────────────────────────────────
nky_holidays = [
    # 2023
    ("2023-01-02", "Mon", "New Year Bank Holiday"),
    ("2023-01-03", "Tue", "New Year Bank Holiday"),
    ("2023-01-09", "Mon", "Coming of Age Day"),
    ("2023-02-23", "Thu", "Emperor's Birthday"),
    ("2023-03-21", "Tue", "Vernal Equinox Day"),
    ("2023-04-29", "Sat", "Showa Day"),  # weekend
    ("2023-05-03", "Wed", "Constitution Memorial Day"),
    ("2023-05-04", "Thu", "Greenery Day"),
    ("2023-05-05", "Fri", "Children's Day"),
    ("2023-07-17", "Mon", "Marine Day"),
    ("2023-08-11", "Fri", "Mountain Day"),
    ("2023-09-18", "Mon", "Respect for the Aged Day"),
    ("2023-10-09", "Mon", "Sports Day"),
    ("2023-11-03", "Fri", "Culture Day"),
    ("2023-11-23", "Thu", "Labour Thanksgiving Day"),
    # 2024
    ("2024-01-01", "Mon", "New Year's Day"),
    ("2024-01-02", "Tue", "New Year Bank Holiday"),
    ("2024-01-03", "Wed", "New Year Bank Holiday"),
    ("2024-01-08", "Mon", "Coming of Age Day"),
    ("2024-02-12", "Mon", "National Foundation Day (observed)"),
    ("2024-02-23", "Fri", "Emperor's Birthday"),
    ("2024-03-20", "Wed", "Vernal Equinox Day"),
    ("2024-04-29", "Mon", "Showa Day"),
    ("2024-05-03", "Fri", "Constitution Memorial Day"),
    ("2024-05-06", "Mon", "Children's Day (observed)"),
    ("2024-07-15", "Mon", "Marine Day"),
    ("2024-08-12", "Mon", "Mountain Day (observed)"),
    ("2024-09-16", "Mon", "Respect for the Aged Day"),
    ("2024-09-23", "Mon", "Autumnal Equinox Day (observed)"),
    ("2024-10-14", "Mon", "Sports Day"),
    ("2024-11-04", "Mon", "Culture Day (observed)"),
    ("2024-12-31", "Tue", "New Year's Eve (market closed)"),
    # 2025
    ("2025-01-01", "Wed", "New Year's Day"),
    ("2025-01-02", "Thu", "New Year Bank Holiday"),
    ("2025-01-03", "Fri", "New Year Bank Holiday"),
    ("2025-01-13", "Mon", "Coming of Age Day"),
    ("2025-02-11", "Tue", "National Foundation Day"),
    ("2025-02-24", "Mon", "Emperor's Birthday (observed)"),
    ("2025-03-20", "Thu", "Vernal Equinox Day"),
    ("2025-04-29", "Tue", "Showa Day"),
    ("2025-05-05", "Mon", "Children's Day"),
    ("2025-05-06", "Tue", "Greenery Day (observed)"),
    ("2025-07-21", "Mon", "Marine Day"),
    ("2025-08-11", "Mon", "Mountain Day"),
    ("2025-09-15", "Mon", "Respect for the Aged Day"),
    ("2025-09-23", "Tue", "Autumnal Equinox Day"),
    ("2025-10-13", "Mon", "Sports Day"),
    ("2025-11-03", "Mon", "Culture Day"),
    ("2025-11-24", "Mon", "Labour Thanksgiving Day (observed)"),
    ("2025-12-31", "Wed", "New Year's Eve (market closed)"),
    # 2026
    ("2026-01-01", "Thu", "New Year's Day"),
    ("2026-01-02", "Fri", "New Year Bank Holiday"),
    ("2026-01-12", "Mon", "Coming of Age Day"),
    ("2026-02-11", "Wed", "National Foundation Day"),
]

# ──────────────────────────────────────────────
# SGX — STI_Holidays (Singapore)
# ──────────────────────────────────────────────
sti_holidays = [
    # 2023
    ("2023-01-23", "Mon", "Lunar New Year Day 1 (observed)"),
    ("2023-01-24", "Tue", "Lunar New Year Day 2"),
    ("2023-04-07", "Fri", "Good Friday"),
    ("2023-04-22", "Sat", "Hari Raya Puasa"),  # weekend
    ("2023-05-01", "Mon", "Labour Day"),
    ("2023-06-02", "Fri", "Vesak Day"),
    ("2023-06-29", "Thu", "Hari Raya Haji"),
    ("2023-08-09", "Wed", "National Day"),
    ("2023-11-13", "Mon", "Deepavali (observed)"),
    ("2023-12-25", "Mon", "Christmas Day"),
    # 2024
    ("2024-01-01", "Mon", "New Year's Day"),
    ("2024-02-12", "Mon", "Lunar New Year Day 2 (observed)"),
    ("2024-03-29", "Fri", "Good Friday"),
    ("2024-04-10", "Wed", "Hari Raya Puasa"),
    ("2024-05-01", "Wed", "Labour Day"),
    ("2024-05-22", "Wed", "Vesak Day"),
    ("2024-06-17", "Mon", "Hari Raya Haji"),
    ("2024-08-09", "Fri", "National Day"),
    ("2024-11-01", "Fri", "Deepavali"),
    ("2024-12-25", "Wed", "Christmas Day"),
    # 2025
    ("2025-01-01", "Wed", "New Year's Day"),
    ("2025-01-29", "Wed", "Lunar New Year Day 1"),
    ("2025-01-30", "Thu", "Lunar New Year Day 2"),
    ("2025-03-31", "Mon", "Hari Raya Puasa"),
    ("2025-04-18", "Fri", "Good Friday"),
    ("2025-05-01", "Thu", "Labour Day"),
    ("2025-05-12", "Mon", "Vesak Day"),
    ("2025-06-06", "Fri", "Hari Raya Haji"),
    ("2025-08-09", "Sat", "National Day"),  # weekend
    ("2025-10-20", "Mon", "Deepavali"),
    ("2025-12-25", "Thu", "Christmas Day"),
    # 2026
    ("2026-01-01", "Thu", "New Year's Day"),
    ("2026-02-17", "Tue", "Lunar New Year Day 1"),
    ("2026-02-18", "Wed", "Lunar New Year Day 2"),
]

# ──────────────────────────────────────────────
# Build and save Trading_Holidays.xlsx
# ──────────────────────────────────────────────
def make_df(data):
    df = pd.DataFrame(data, columns=['Date', 'Day', 'Holiday'])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.drop_duplicates(subset='Date').sort_values('Date').reset_index(drop=True)
    return df

all_sheets = {
    'NIFTY_Holidays':  make_df(nifty_holidays),
    'DJ_Holidays':     make_df(usa_holidays),
    'SP_Holidays':     make_df(usa_holidays),      # same schedule as DJ
    'DAX_Holidays':    make_df(dax_holidays),
    'UKX_Holidays':    make_df(ukx_holidays),
    'HSI_Holidays':    make_df(hsi_holidays),
    'SHCOMP_Holidays': make_df(shcomp_holidays),
    'TWSE_Holidays':   make_df(twse_holidays),
    'NKY_Holidays':    make_df(nky_holidays),
    'STI_Holidays':    make_df(sti_holidays),
}

output_path = os.path.join(input_fp, 'Trading_Holidays.xlsx')
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    for sheet_name, df in all_sheets.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f'Created: {output_path}')
print(f'Sheets:')
for name, df in all_sheets.items():
    print(f'  {name}: {len(df)} holidays')
