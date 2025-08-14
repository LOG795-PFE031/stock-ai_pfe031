import { isWeekend, subDays, format } from 'date-fns';

/**
 * Gets a date range covering the last `days` business days ending today.
 *
 * @param {number} [days=30] - Number of business days to include.
 * @returns {{ startDate: string, endDate: string }} - Formatted start and end dates.
 */
export function getBusinessDateRange(days: number = 30) : { startDate: string; endDate: string } {
  let end = new Date();

  // Adjust end to the most recent business day if today is a weekend 
   while (isWeekend(end))
    end = subDays(end, 1)

  let current = new Date(end);
  let count = 0;

  while (count != days - 1) {
    current = subDays(current, 1);
    if (!isWeekend(current)) {
      count++;
    }
  }

  return {
    startDate: format(current, 'yyyy-MM-dd'),
    endDate: format(end, 'yyyy-MM-dd'),
  };
}