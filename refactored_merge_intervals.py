def merge_overlapping_ranges(interval_list):
    """
    Combines overlapping ranges in a list of intervals.
    
    This function takes a list of intervals where each interval is represented
    as [start_point, end_point] and merges any overlapping intervals.
    
    Parameters:
        interval_list (List[List[int]]): A list of intervals to merge
        
    Returns:
        List[List[int]]: A new list with merged intervals
    """
    # Handle empty input
    if not interval_list or len(interval_list) == 0:
        return []
        
    # Create a copy to avoid modifying the input
    sorted_intervals = sorted(interval_list, key=lambda interval: interval[0])
    
    # Initialize output with the first interval
    merged_output = []
    current_interval = sorted_intervals[0].copy()  # Create a copy of the first interval
    merged_output.append(current_interval)
    
    # Process remaining intervals
    for next_start, next_end in sorted_intervals[1:]:
        # Get the last interval in our result
        last_end = merged_output[-1][1]
        
        # Check if current interval overlaps with the last merged interval
        if next_start <= last_end:
            # We have an overlap - update the end point if needed
            merged_output[-1][1] = max(last_end, next_end)
        else:
            # No overlap - add this interval to the result
            merged_output.append([next_start, next_end])
    
    return merged_output

# Alternative implementation that uses a different approach
def merge_intervals_alternative(intervals):
    """Alternative implementation using a different approach"""
    # Handle edge cases
    if not intervals:
        return []
    
    # Sort by start time
    intervals = sorted(intervals, key=lambda x: x[0])
    
    # Initialize result list
    output = []
    
    # Track the current merged interval
    current_start = intervals[0][0]
    current_end = intervals[0][1]
    
    # Process all intervals
    for interval in intervals[1:]:
        start, end = interval
        
        # If current interval overlaps with the tracked one
        if start <= current_end:
            # Extend the end of the tracked interval if needed
            current_end = max(current_end, end)
        else:
            # No overlap, add the tracked interval to result and start tracking the current one
            output.append([current_start, current_end])
            current_start, current_end = start, end
    
    # Add the last tracked interval
    output.append([current_start, current_end])
    
    return output
