from datetime import datetime, timedelta
from typing import Optional, Tuple

def get_date_range(start_date: Optional[str], end_date: Optional[str]) -> Tuple[datetime, datetime]:
    today = datetime.now()
    if end_date:
        end = datetime.fromisoformat(end_date)
    else:
        end = today
    if start_date:
        start = datetime.fromisoformat(start_date)
    else:
        start = end - timedelta(days=7)
    return start, end 
