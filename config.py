"""
Configuration file for Signal Generator
Put your API keys here (this file should be in .gitignore)
"""

# REQUIRED: Polygon API Key (get from polygon.io)
POLYGON_API_KEY = "your_polygon_api_key_here"

# OPTIONAL: Supabase credentials (leave as-is if not using)
SUPABASE_URL = "your_supabase_url_here"
SUPABASE_KEY = "your_supabase_key_here"
SUPABASE_TABLE = "trading_signals"

# Active timeframes to monitor
ACTIVE_TIMEFRAMES = ['5T', '15T', '30T']