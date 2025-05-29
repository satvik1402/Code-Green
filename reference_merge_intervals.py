def merge(intervals):
    """
    Merge overlapping intervals.
    
    Args:
        intervals: List of interval pairs [start, end]
    Returns:
        List of merged intervals
    """
    # Edge case: empty input
    if not intervals:
        return []
    
    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])
    
    # Initialize result with first interval
    result = [intervals[0]]
    
    # Iterate through intervals
    for i in range(1, len(intervals)):
        current = intervals[i]
        previous = result[-1]
        
        # If current interval overlaps with previous interval
        if current[0] <= previous[1]:
            # Merge them by updating the end of the previous interval
            previous[1] = max(previous[1], current[1])
        else:
            # No overlap, add current interval to result
            result.append(current)
    
    return result
