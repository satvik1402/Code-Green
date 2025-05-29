from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from CodePlagueDetector import db
from CodePlagueDetector.models import Submission, PlagiarismReference
from CodePlagueDetector.plagiarism_detector import PlagiarismDetector
from CodePlagueDetector.dsa_problems import DSA_PROBLEMS
import json
import logging

# Create blueprint
main = Blueprint('main', __name__)

# Initialize plagiarism detector
detector = PlagiarismDetector()

@main.route('/')
def index():
    """Display the main page with DSA problems selection"""
    return render_template('index.html', problems=DSA_PROBLEMS)

@main.route('/submit_code', methods=['POST'])
def submit_code():
    try:
        if request.method == 'POST':
            problem_id = request.form.get('problem_id')
            code = request.form.get('code')
            fast_typing_detected = request.form.get('fast_typing_detected') == 'true'
            language = request.form.get('language', 'python')
            
            if not problem_id or not code:
                flash('Missing problem ID or code', 'error')
                return redirect(url_for('main.index'))
                
            # Check if fast typing was detected
            if fast_typing_detected:
                flash('Fast typing or code pasting detected. This has been recorded.', 'warning')
                # Could log this to a database in a real system
            
            # Get the selected problem
            selected_problem = next((p for p in DSA_PROBLEMS if p['id'] == problem_id), None)
            if not selected_problem:
                flash('Invalid problem selected.', 'error')
                return redirect(url_for('main.index'))
        
            # Create submission record
            submission = Submission(
                problem_id=problem_id,
                code=code,
                language=language,
                fast_typing_detected=fast_typing_detected
            )
        
            # Get reference solutions for this problem
            reference_solutions = PlagiarismReference.query.filter_by(
                problem_id=problem_id,
                language=language
            ).all()
        
            # If no reference solutions exist, add some default ones
            if not reference_solutions:
                _add_default_reference_solutions(problem_id, language)
                reference_solutions = PlagiarismReference.query.filter_by(
                    problem_id=problem_id,
                    language=language
                ).all()
            
            # To improve detection, we'll add the current submission as a temporary reference solution
            # This helps detect if future submissions are similar to this one (likely AI-generated)
            # We'll create a temporary reference without saving it to the database
            temp_reference = PlagiarismReference(
                problem_id=problem_id,
                solution_code=code,
                solution_name='Current Submission',
                language=language
            )
            
            # Add it to our reference list for comparison
            reference_solutions = list(reference_solutions) + [temp_reference]
        
            # Run test cases first
            test_result = _run_test_cases(code, problem_id)
            passed_tests = test_result.get('passed', False)
            test_details = test_result.get('details', 'No test details available')
            
            # Perform plagiarism detection only if tests pass
            logging.info(f"Analyzing code for problem {problem_id} with {len(reference_solutions)} reference solutions")
            logging.info(f"Test cases passed: {passed_tests}")
        
            # Check if we have reference solutions
            reference_codes = [ref.solution_code for ref in reference_solutions]
            if len(reference_codes) <= 1:  # Only our temp solution
                logging.warning(f"No reference solutions found for problem {problem_id}!")
                flash('No reference solutions available for this problem. Using AI detection heuristics.', 'warning')
                
                # Add a placeholder solution with common patterns for each problem type
                # This helps detect AI-generated solutions even without specific references
                generic_pattern = f"""def {selected_problem['function_signature'].split('(')[0]}(...):
    # Initialize variables
    # Check for edge cases
    # Main algorithm logic
    # Time complexity: O(n)
    # Space complexity: O(n)
    # Return result"""
                
                # Add the generic pattern to our references
                reference_codes.append(generic_pattern)
        
            # Store test results in the submission
            submission.test_passed = passed_tests
            submission.test_details = test_details
            submission.fast_typing_detected = fast_typing_detected
            
            # Skip plagiarism detection if tests fail
            if not passed_tests:
                # Save submission with test failure details
                db.session.add(submission)
                db.session.commit()
                
                # Show the test failure results
                return render_template('results.html',
                                      submission=submission,
                                      results={'feedback': [{
                                          'type': 'error',
                                          'message': 'Test cases failed. Please fix your code before plagiarism detection.',
                                          'severity': 'high'
                                      }]},
                                      submissions=[],
                                      problem_id=problem_id,
                                      passed_tests=passed_tests)
        
            # Log a sample of submitted code (first 100 chars)
            code_sample = code[:100] + '...' if len(code) > 100 else code
            logging.info(f"Submitted code sample: {code_sample}")
        
            # First, test the code against test cases to ensure it's valid
            # This helps verify the user didn't submit code for a different problem
            test_result = _run_test_cases(code, problem_id)
        
            if not test_result['passed']:
                # If tests don't pass, we'll still analyze but flag it
                logging.warning(f"Submitted code failed test cases for problem {problem_id}")
                flash('Your code failed some test cases. It may not be a valid solution.', 'warning')
        
            # Get reference solution codes
            reference_codes = [ref.solution_code for ref in reference_solutions]
        
            # Analyze for plagiarism, passing the test status
            results = detector.analyze_code(code, reference_codes, problem_id=problem_id, test_passed=passed_tests)
        
            # Prevent showing exact match for refactored code
            # If the code has renamed variables or refactored structure but still passes tests,
            # it shouldn't be flagged as an exact match
            if results.get('exact_match_detected', False) and results.get('refactoring_detected', False):
                results['exact_match_detected'] = False
                results['feedback'] = results.get('feedback', '') + '\nCode appears to be refactored from a reference solution.'
                
            # Update submission with results
            submission.exact_match = results.get('exact_match_detected', False)
            submission.variable_renaming = results.get('variable_renaming_detected', False)
            submission.structural_similarity = results.get('structural_similarity_detected', False)
            submission.comment_similarity = results.get('comment_similarity_detected', False)
            submission.semantic_similarity = results.get('similarity_score', 0.0)
            submission.feedback_data = json.dumps(results)
            
            # Save to database
            db.session.add(submission)
            db.session.commit()
            
            # Log final results
            logging.info(f"Analysis completed for submission - " +
                         f"Exact match: {submission.exact_match}, " +
                         f"Variable renaming: {submission.variable_renaming}, " +
                         f"Structural similarity: {submission.structural_similarity}, " +
                         f"Comment similarity: {submission.comment_similarity}, " +
                         f"Semantic similarity: {submission.semantic_similarity:.4f}")
            
            return redirect(url_for('main.results', submission_id=submission.id))
            
    except Exception as e:
        logging.error(f"Error in submit_code: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        flash('An error occurred while analyzing your code. Please try again.', 'error')
        
        # Create a minimal submission record to show error details
        if 'submission' in locals() and submission:
            # Use the existing submission if we have one
            try:
                submission.feedback_data = json.dumps({'feedback': [{
                    'type': 'error',
                    'message': f'Error analyzing code: {str(e)}',
                    'severity': 'high'
                }]})
                db.session.add(submission)
                db.session.commit()
                return redirect(url_for('main.results', submission_id=submission.id))
            except Exception as inner_e:
                logging.error(f"Error handling exception gracefully: {str(inner_e)}")
        
        return redirect(url_for('main.index'))

@main.route('/results/<int:submission_id>')
def results(submission_id):
    """Display plagiarism detection results"""
    submission = Submission.query.get_or_404(submission_id)
    
    # Parse feedback data
    feedback = []
    if submission.feedback_data:
        try:
            feedback = json.loads(submission.feedback_data)
        except:
            feedback = []
    
    # Calculate plagiarism percentage with weighted importance
    # Give higher weights to more important indicators
    plagiarism_weights = {
        'exact_match': 1.0,          # Exact match is definitive proof (100%)
        'variable_renaming': 0.8,    # Variable renaming is strong evidence (80%)
        'structural_similarity': 0.7, # Similar structure with same algorithm (70%)
        'comment_similarity': 0.6,   # Similar comments (60%)
        'semantic_similarity': 0.9,  # Semantic similarity (90%)
        'fast_typing': 0.5           # Fast typing is a weak indicator (50%)
    }
    
    # Determine if each type of plagiarism is detected
    indicators = {
        'exact_match': submission.exact_match,
        'variable_renaming': submission.variable_renaming,
        'structural_similarity': submission.structural_similarity,
        'comment_similarity': submission.comment_similarity,
        'semantic_similarity': submission.semantic_similarity > 0.6,  # Lower threshold to 60%
        'fast_typing': submission.fast_typing_detected
    }
    
    # Calculate weighted score
    weighted_score = 0
    total_weight = 0
    active_indicators = 0
    detected_types = []
    
    # Add exact match weight if detected
    if indicators['exact_match']:
        weighted_score += plagiarism_weights['exact_match']
        total_weight += plagiarism_weights['exact_match']
        active_indicators += 1
        detected_types.append('exact_match')
    
    # Add variable renaming weight if detected
    elif indicators['variable_renaming'] and not indicators['exact_match']:
        weighted_score += plagiarism_weights['variable_renaming']
        total_weight += plagiarism_weights['variable_renaming']
        active_indicators += 1
        detected_types.append('variable_renaming')
    
    # Add structural similarity weight if detected and not already covered
    elif indicators['structural_similarity'] and not indicators['exact_match'] and not indicators['variable_renaming']:
        weighted_score += plagiarism_weights['structural_similarity']
        total_weight += plagiarism_weights['structural_similarity']
        active_indicators += 1
        detected_types.append('structural_similarity')
    
    # Add comment similarity weight if detected
    if indicators['comment_similarity']:
        weighted_score += plagiarism_weights['comment_similarity']
        total_weight += plagiarism_weights['comment_similarity']
        active_indicators += 1
        detected_types.append('comment_similarity')
    
    # Add semantic similarity weight (scaled by actual similarity)
    if indicators['semantic_similarity']:
        semantic_weight = plagiarism_weights['semantic_similarity'] * min(1.0, submission.semantic_similarity / 0.8)
        weighted_score += semantic_weight
        total_weight += plagiarism_weights['semantic_similarity']
        active_indicators += 1
        detected_types.append('semantic_similarity')
    
    # Add fast typing weight if detected
    if indicators['fast_typing']:
        weighted_score += plagiarism_weights['fast_typing']
        total_weight += plagiarism_weights['fast_typing']
        active_indicators += 1
        detected_types.append('fast_typing')
    
    # Calculate final score with diminishing returns for multiple indicators
    # This ensures refactored code gets a lower score
    if active_indicators > 1:
        # Apply a discount factor for refactored code (more indicators but not exact match)
        if 'exact_match' not in detected_types and active_indicators >= 2:
            weighted_score *= 0.8  # 20% discount for likely refactored code
        
        # Further discount if only semantic similarity is high but structure is different
        if 'semantic_similarity' in detected_types and 'structural_similarity' not in detected_types \
           and 'variable_renaming' not in detected_types and 'exact_match' not in detected_types:
            weighted_score *= 0.7  # 30% discount for conceptually similar but structurally different code
    
    plagiarism_percentage = (weighted_score / total_weight) * 100 if total_weight > 0 else 0
    # Cap at 100%
    plagiarism_percentage = min(100, plagiarism_percentage)
    originality_percentage = 100 - plagiarism_percentage
    
    # Get problem details
    problem = next((p for p in DSA_PROBLEMS if p['id'] == submission.problem_id), None)
    
    results_data = {
        'submission': submission,
        'problem': problem,
        'feedback': feedback,
        'plagiarism_percentage': round(plagiarism_percentage, 1),
        'originality_percentage': round(originality_percentage, 1),
        'semantic_similarity_percentage': round(submission.semantic_similarity * 100, 1)
    }
    
    return render_template('results.html', **results_data)

@main.route('/api/chart-data/<int:submission_id>')
def chart_data(submission_id):
    """API endpoint for chart data"""
    submission = Submission.query.get_or_404(submission_id)
    
    # Calculate plagiarism percentage
    plagiarism_indicators = [
        submission.exact_match,
        submission.variable_renaming,
        submission.structural_similarity,
        submission.comment_similarity,
        submission.semantic_similarity > 0.8
    ]
    
    plagiarism_count = sum(plagiarism_indicators)
    total_checks = len(plagiarism_indicators)
    plagiarism_percentage = (plagiarism_count / total_checks) * 100
    originality_percentage = 100 - plagiarism_percentage
    
    return jsonify({
        'plagiarism': round(plagiarism_percentage, 1),
        'originality': round(originality_percentage, 1)
    })

def _run_test_cases(code, problem_id):
    """Run test cases for the submitted code to verify it's a correct solution
    Returns a dict with 'passed' (bool) and 'details' (str) keys
    """
    try:
        # Create a safe execution environment
        local_vars = {}
        
        # Execute the user's code in the environment
        exec(code, {}, local_vars)
        
        # Get the function based on problem_id
        expected_function_name = next((p['function_signature'].split('(')[0].strip('def ') 
                                   for p in DSA_PROBLEMS if p['id'] == problem_id), None)
        
        # Extract all function names from the code
        function_names = [name for name, obj in local_vars.items() if callable(obj)]
        
        # If no functions found
        if not function_names:
            return {'passed': False, 'details': f"No functions found in your code. Expected function: {expected_function_name}"}
        
        # First, try exact match
        if expected_function_name in function_names:
            function_name = expected_function_name
        else:
            # Check if any function name has the same pattern (could be renamed)
            if problem_id == 'trapping_rain_water' and any(name in ['trap', 'trap_water', 'trapping_rain_water', 'get_water', 'calculate_water'] for name in function_names):
                function_name = next(name for name in function_names if name in ['trap', 'trap_water', 'trapping_rain_water', 'get_water', 'calculate_water'])
            elif problem_id == 'two_sum' and any(name in ['two_sum', 'twoSum', 'find_pair', 'find_two_sum'] for name in function_names):
                function_name = next(name for name in function_names if name in ['two_sum', 'twoSum', 'find_pair', 'find_two_sum'])
            elif len(function_names) == 1:
                # If only one function is defined, use it regardless of name
                function_name = function_names[0]
                logging.info(f"Using function {function_name} instead of expected {expected_function_name}")
            else:
                return {'passed': False, 'details': f"Function {expected_function_name} not found in your code. Found functions: {', '.join(function_names)}"}
        
        # Check if the function exists in local variables
        if function_name not in local_vars:
            return {'passed': False, 'details': f"Function {function_name} not found in your code"}
        
        # Get the function from the environment
        user_func = local_vars[function_name]
        
        # Run tests based on problem_id
        if problem_id == 'two_sum':
            test_cases = [
                {'input': {'nums': [2, 7, 11, 15], 'target': 9}, 'expected': [0, 1]},
                {'input': {'nums': [3, 2, 4], 'target': 6}, 'expected': [1, 2]}
            ]
        elif problem_id == 'trapping_rain_water':
            test_cases = [
                {'input': {'height': [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]}, 'expected': 6},
                {'input': {'height': [4, 2, 0, 3, 2, 5]}, 'expected': 9}
            ]
        elif problem_id == 'merge_intervals':
            test_cases = [
                {'input': {'intervals': [[1, 3], [2, 6], [8, 10], [15, 18]]}, 'expected': [[1, 6], [8, 10], [15, 18]]},
                {'input': {'intervals': [[1, 4], [4, 5]]}, 'expected': [[1, 5]]}
            ]
        elif problem_id == 'maximum_subarray':
            test_cases = [
                {'input': {'nums': [-2, 1, -3, 4, -1, 2, 1, -5, 4]}, 'expected': 6},
                {'input': {'nums': [5, 4, -1, 7, 8]}, 'expected': 23}
            ]
        elif problem_id == 'valid_parentheses':
            test_cases = [
                {'input': {'s': "()[]{}"}, 'expected': True},
                {'input': {'s': "([)]"}, 'expected': False}
            ]
        elif problem_id == 'longest_substring_no_repeat':
            test_cases = [
                {'input': {'s': "abcabcbb"}, 'expected': 3},
                {'input': {'s': "pwwkew"}, 'expected': 3}
            ]
        elif problem_id == 'kth_largest_element':
            test_cases = [
                {'input': {'nums': [3, 2, 1, 5, 6, 4], 'k': 2}, 'expected': 5},
                {'input': {'nums': [3, 2, 3, 1, 2, 4, 5, 5, 6], 'k': 4}, 'expected': 4}
            ]
        else:
            # For other problems, we'll skip testing for now
            return {'passed': True, 'details': "Test cases not implemented for this problem"}
            
        # Run the tests
        for i, test in enumerate(test_cases):
            result = user_func(**test['input'])
            
            # For list results, convert to sorted tuples for comparison
            if isinstance(result, list) and isinstance(test['expected'], list):
                if isinstance(result[0], list) and isinstance(test['expected'][0], list):
                    # For nested lists like merge_intervals
                    result_set = {tuple(map(tuple, sorted(result)))}
                    expected_set = {tuple(map(tuple, sorted(test['expected'])))}
                else:
                    # For simple lists like two_sum
                    result_set = {tuple(sorted(result))}
                    expected_set = {tuple(sorted(test['expected']))}
                    
                if result_set != expected_set:
                    return {'passed': False, 'details': f"Test {i+1} failed: expected {test['expected']}, got {result}"}
            else:
                # Direct comparison for non-list results
                if result != test['expected']:
                    return {'passed': False, 'details': f"Test {i+1} failed: expected {test['expected']}, got {result}"}
        
        return {'passed': True, 'details': "All tests passed"}
        
    except Exception as e:
        logging.error(f"Error running test cases: {e}")
        return {'passed': False, 'details': f"Error executing code: {str(e)}"}

def _add_default_reference_solutions(problem_id, language='python'):
    """Add default reference solutions for testing purposes"""
    if language != 'python':
        return
    
    # Sample reference solutions for different problems
    reference_solutions = {
        'trapping_rain_water': [
            '''def trap(height):
    if not height:
        return 0
        
    left, right = 0, len(height) - 1
    left_max = right_max = 0
    water = 0
    
    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1
            
    return water''',
            '''def trap(height):
    # Edge case
    if not height or len(height) < 3:
        return 0
    
    # Calculate left max array
    left_max = [0] * len(height)
    left_max[0] = height[0]
    for i in range(1, len(height)):
        left_max[i] = max(left_max[i-1], height[i])
    
    # Calculate right max array
    right_max = [0] * len(height)
    right_max[-1] = height[-1]
    for i in range(len(height)-2, -1, -1):
        right_max[i] = max(right_max[i+1], height[i])
    
    # Calculate trapped water
    water = 0
    for i in range(len(height)):
        water += min(left_max[i], right_max[i]) - height[i]
    
    return water'''
        ],
        'two_sum': [
            '''def two_sum(nums, target):
    # Dictionary to store numbers: index
    num_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    return []''',
            '''def two_sum(nums, target):
    # Use hash map for O(n) solution
    hash_map = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in hash_map:
            return [hash_map[complement], i]
        hash_map[num] = i
    return []''',
            '''def two_sum(nums, target):
    # Brute force approach O(n^2)
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []''',
            '''def two_sum(nums, target):
    """
    Find two numbers that add up to target
    Args:
        nums: List of integers
        target: Target sum
    Returns:
        List of indices
    """
    seen = {}
    for index, value in enumerate(nums):
        remaining = target - value
        if remaining in seen:
            return [seen[remaining], index]
        seen[value] = index
    return []'''
        ],
        'merge_intervals': [
            '''def merge(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        if current[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], current[1])
        else:
            merged.append(current)
    return merged'''
        ],
        'reverse_linked_list': [
            '''def reverse_list(head):
    prev = None
    current = head
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
    return prev'''
        ]
    }
    
    # Add additional reference solutions for more problems
    if problem_id == 'detect_cycle_graph' and not PlagiarismReference.query.filter_by(problem_id=problem_id).first():
        solutions = [
            '''def has_cycle(graph):
    # Use depth-first search to detect cycles
    visited = set()
    rec_stack = set()
    
    def dfs(node):
        # Mark current node as visited and add to recursion stack
        visited.add(node)
        rec_stack.add(node)
        
        # Visit all neighbors
        for neighbor in graph.get(node, []):
            # If not visited, check if cycle exists in the subtree
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            # If already in recursion stack, there's a cycle
            elif neighbor in rec_stack:
                return True
                
        # Remove from recursion stack
        rec_stack.remove(node)
        return False
    
    # Check all nodes (to handle disconnected components)
    for node in graph:
        if node not in visited:
            if dfs(node):
                return True
    
    return False''',
            '''def has_cycle(graph):
    visited = {}  # 0: not visited, 1: in progress, 2: completed
    
    def dfs(node):
        if node in visited:
            return visited[node] == 1  # If in progress, cycle detected
        
        visited[node] = 1  # Mark as in progress
        
        for neighbor in graph.get(node, []):
            if dfs(neighbor):
                return True
                
        visited[node] = 2  # Mark as completed
        return False
    
    for node in graph:
        if node not in visited:
            if dfs(node):
                return True
                
    return False'''
        ]
        for i, solution in enumerate(solutions):
            ref = PlagiarismReference(
                problem_id=problem_id,
                solution_code=solution,
                solution_name=f'Reference Solution {i+1}',
                language=language
            )
            db.session.add(ref)
        db.session.commit()
        
    elif problem_id == 'kth_largest_element' and not PlagiarismReference.query.filter_by(problem_id=problem_id).first():
        solutions = [
            '''def find_kth_largest(nums, k):
    # Using quickselect algorithm (average O(n) time complexity)
    def partition(left, right, pivot_index):
        pivot = nums[pivot_index]
        # Move pivot to end
        nums[pivot_index], nums[right] = nums[right], nums[pivot_index]
        
        # Move all elements smaller than pivot to the left
        store_index = left
        for i in range(left, right):
            if nums[i] < pivot:
                nums[store_index], nums[i] = nums[i], nums[store_index]
                store_index += 1
                
        # Move pivot to its final place
        nums[right], nums[store_index] = nums[store_index], nums[right]
        return store_index
    
    def quick_select(left, right, k_smallest):
        # If list contains only one element, return that element
        if left == right:
            return nums[left]
            
        # Select a random pivot_index between left and right
        import random
        pivot_index = random.randint(left, right)
        
        # Find the pivot position in a sorted list
        pivot_index = partition(left, right, pivot_index)
        
        # If pivot is in its final sorted position
        if k_smallest == pivot_index:
            return nums[k_smallest]
        # If there are more small elements than k_smallest, go left
        elif k_smallest < pivot_index:
            return quick_select(left, pivot_index - 1, k_smallest)
        # Otherwise, go right
        else:
            return quick_select(pivot_index + 1, right, k_smallest)
    
    # Kth largest is (n - k)th smallest
    return quick_select(0, len(nums) - 1, len(nums) - k)
''',
            '''def find_kth_largest(nums, k):
    # Using Python's built-in sorting
    return sorted(nums, reverse=True)[k-1]
''',
            '''def find_kth_largest(nums, k):
    # Using a heap to efficiently find the kth largest element
    import heapq
    # Create a min-heap of size k
    heap = []
    
    for num in nums:
        # If heap has less than k elements, add the element
        if len(heap) < k:
            heapq.heappush(heap, num)
        # Otherwise, if current element is larger than the smallest in heap
        elif num > heap[0]:
            # Replace the smallest element
            heapq.heapreplace(heap, num)
    
    # Return the smallest element in the heap (kth largest overall)
    return heap[0]
'''
        ]
        for i, solution in enumerate(solutions):
            ref = PlagiarismReference(
                problem_id=problem_id,
                solution_code=solution,
                solution_name=f'Reference Solution {i+1}',
                language=language
            )
            db.session.add(ref)
        db.session.commit()
    
    elif problem_id == 'valid_parentheses' and not PlagiarismReference.query.filter_by(problem_id=problem_id).first():
        solutions = [
            '''def is_valid(s):
    # Using a stack to keep track of opening brackets
    stack = []
    # Mapping of closing to opening brackets
    brackets_map = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        # If it's an opening bracket, push to stack
        if char in '({[':
            stack.append(char)
        # If it's a closing bracket
        elif char in ')}]':
            # If stack is empty or doesn't match, return False
            if not stack or brackets_map[char] != stack.pop():
                return False
    
    # If stack is empty, all brackets were matched
    return len(stack) == 0
''',
            '''def is_valid(s):
    # Initialize a stack
    stack = []
    
    # Iterate through each character in the string
    for c in s:
        # If opening bracket, push to stack
        if c == '(' or c == '{' or c == '[':
            stack.append(c)
        # If closing bracket, check if it matches the top of the stack
        else:
            if not stack:
                return False
            
            top = stack.pop()
            
            if c == ')' and top != '(':
                return False
            if c == '}' and top != '{':
                return False
            if c == ']' and top != '[':
                return False
    
    # If stack is empty, all brackets were matched
    return len(stack) == 0
'''
        ]
        for i, solution in enumerate(solutions):
            ref = PlagiarismReference(
                problem_id=problem_id,
                solution_code=solution,
                solution_name=f'Reference Solution {i+1}',
                language=language
            )
            db.session.add(ref)
        db.session.commit()
        
    elif problem_id == 'binary_tree_level_order' and not PlagiarismReference.query.filter_by(problem_id=problem_id).first():
        solutions = [
            '''def level_order(root):
    # Handle empty tree
    if not root:
        return []
    
    # Initialize result list and queue for BFS
    result = []
    queue = [root]
    
    while queue:
        # Get the number of nodes at current level
        level_size = len(queue)
        level_nodes = []
        
        # Process all nodes at current level
        for _ in range(level_size):
            node = queue.pop(0)  # Dequeue node
            level_nodes.append(node.val)  # Add node value to current level
            
            # Enqueue children
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        # Add current level to result
        result.append(level_nodes)
    
    return result
''',
            '''def level_order(root):
    # Use BFS with a queue to traverse level by level
    from collections import deque
    
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level = []
        level_size = len(queue)
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
                
        result.append(level)
    
    return result
'''
        ]
        for i, solution in enumerate(solutions):
            ref = PlagiarismReference(
                problem_id=problem_id,
                solution_code=solution,
                solution_name=f'Reference Solution {i+1}',
                language=language
            )
            db.session.add(ref)
        db.session.commit()
        
    elif problem_id == 'word_ladder' and not PlagiarismReference.query.filter_by(problem_id=problem_id).first():
        solutions = [
            '''def ladder_length(begin_word, end_word, word_list):
    # Convert word_list to a set for O(1) lookups
    word_set = set(word_list)
    
    # If end_word is not in the word_list, no transformation sequence exists
    if end_word not in word_set:
        return 0
    
    # Initialize queue for BFS
    from collections import deque
    queue = deque([(begin_word, 1)])  # (word, level)
    visited = set([begin_word])
    
    while queue:
        current_word, level = queue.popleft()
        
        # Try all possible one-letter transformations
        for i in range(len(current_word)):
            # Try replacing each character with all letters
            for c in 'abcdefghijklmnopqrstuvwxyz':
                # Create a new word with one character changed
                next_word = current_word[:i] + c + current_word[i+1:]
                
                # If we reached the end word, return the level + 1
                if next_word == end_word:
                    return level + 1
                    
                # If the word is in word_list and not visited
                if next_word in word_set and next_word not in visited:
                    visited.add(next_word)
                    queue.append((next_word, level + 1))
    
    # If no transformation sequence exists
    return 0
''',
            '''def ladder_length(begin_word, end_word, word_list):
    # Bidirectional BFS
    if end_word not in word_list:
        return 0
        
    # Create a set for O(1) lookups
    word_set = set(word_list)
    
    # Initialize front and back queues for bidirectional search
    front, back = {begin_word}, {end_word}
    # Initialize visited sets for both directions
    visited_front, visited_back = {begin_word}, {end_word}
    # Length of the transformation sequence
    length = 1
    
    # Continue until either front or back queue is empty
    while front and back:
        # Always expand the smaller queue for efficiency
        if len(front) > len(back):
            front, back = back, front
            visited_front, visited_back = visited_back, visited_front
        
        # Next level of words
        next_level = set()
        
        # Expand all words in the current level
        for word in front:
            # Try all possible one-letter transformations
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    new_word = word[:i] + c + word[i+1:]
                    
                    # If we've reached a word from the other direction
                    if new_word in back:
                        return length + 1
                        
                    # If it's a valid word we haven't visited
                    if new_word in word_set and new_word not in visited_front:
                        next_level.add(new_word)
                        visited_front.add(new_word)
        
        # Move to the next level
        front = next_level
        length += 1
    
    # If no path is found
    return 0
'''
        ]
        for i, solution in enumerate(solutions):
            ref = PlagiarismReference(
                problem_id=problem_id,
                solution_code=solution,
                solution_name=f'Reference Solution {i+1}',
                language=language
            )
            db.session.add(ref)
        db.session.commit()
        
    elif problem_id == 'longest_substring_no_repeat' and not PlagiarismReference.query.filter_by(problem_id=problem_id).first():
        solutions = [
            '''def length_of_longest_substring(s):
    char_dict = {}
    max_length = start = 0
    
    for i, char in enumerate(s):
        # If char is already in the current window, update the start pointer
        if char in char_dict and start <= char_dict[char]:
            start = char_dict[char] + 1
        else:
            # Update max length if current window is longer
            max_length = max(max_length, i - start + 1)
            
        # Update char position
        char_dict[char] = i
        
    return max_length''',
            '''def length_of_longest_substring(s):
    # Sliding window approach
    if not s:
        return 0
        
    n = len(s)
    max_len = 0
    char_set = set()
    left = 0
    
    for right in range(n):
        # If character already in set, remove characters from left until no duplicates
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
            
        # Add current character
        char_set.add(s[right])
        # Update max length
        max_len = max(max_len, right - left + 1)
        
    return max_len'''
        ]
        for i, solution in enumerate(solutions):
            ref = PlagiarismReference(
                problem_id=problem_id,
                solution_code=solution,
                solution_name=f'Reference Solution {i+1}',
                language=language
            )
            db.session.add(ref)
        db.session.commit()
        
    elif problem_id == 'maximum_subarray' and not PlagiarismReference.query.filter_by(problem_id=problem_id).first():
        solutions = [
            '''def max_subarray(nums):
    # Kadane's algorithm for maximum subarray
    if not nums:
        return 0
        
    current_sum = max_sum = nums[0]
    
    for num in nums[1:]:
        # Either start a new subarray or extend the existing one
        current_sum = max(num, current_sum + num)
        # Update max_sum if current_sum is larger
        max_sum = max(max_sum, current_sum)
        
    return max_sum''',
            '''def max_subarray(nums):
    # Dynamic programming approach
    if not nums:
        return 0
        
    # dp[i] represents the maximum subarray sum ending at index i
    dp = [0] * len(nums)
    dp[0] = nums[0]
    
    for i in range(1, len(nums)):
        dp[i] = max(nums[i], dp[i-1] + nums[i])
        
    return max(dp)''',
            '''def max_subarray(nums):
    # Divide and conquer approach
    def divide_and_conquer(start, end):
        # Base case: single element
        if start == end:
            return nums[start]
            
        # Find middle point
        mid = (start + end) // 2
        
        # Find maximum subarray sum in left half
        left_sum = divide_and_conquer(start, mid)
        # Find maximum subarray sum in right half
        right_sum = divide_and_conquer(mid + 1, end)
        
        # Find maximum subarray sum that crosses the middle
        left_max = float('-inf')
        current = 0
        for i in range(mid, start - 1, -1):
            current += nums[i]
            left_max = max(left_max, current)
            
        right_max = float('-inf')
        current = 0
        for i in range(mid + 1, end + 1):
            current += nums[i]
            right_max = max(right_max, current)
            
        # Return the maximum of the three
        cross_sum = left_max + right_max
        return max(left_sum, right_sum, cross_sum)
        
    return divide_and_conquer(0, len(nums) - 1)'''
        ]
        for i, solution in enumerate(solutions):
            ref = PlagiarismReference(
                problem_id=problem_id,
                solution_code=solution,
                solution_name=f'Reference Solution {i+1}',
                language=language
            )
            db.session.add(ref)
        db.session.commit()
        
    elif problem_id == 'clone_graph' and not PlagiarismReference.query.filter_by(problem_id=problem_id).first():
        solutions = [
            '''def clone_graph(node):
    # Handle empty graph
    if not node:
        return None
        
    # Use DFS with a hash map to keep track of cloned nodes
    visited = {}
    
    def dfs(original):
        # If node already visited, return the clone
        if original in visited:
            return visited[original]
            
        # Create a clone of the current node
        clone = Node(original.val, [])
        # Add to visited map
        visited[original] = clone
        
        # Recursively clone all neighbors
        for neighbor in original.neighbors:
            clone.neighbors.append(dfs(neighbor))
            
        return clone
        
    return dfs(node)''',
            '''def clone_graph(node):
    # BFS approach with a queue
    if not node:
        return None
        
    from collections import deque
    
    # Create a clone of the first node
    clone = Node(node.val, [])
    # Map to track cloned nodes
    visited = {node: clone}
    # Queue for BFS
    queue = deque([node])
    
    while queue:
        current = queue.popleft()
        
        # Process all neighbors
        for neighbor in current.neighbors:
            # If neighbor not visited, create a clone and add to queue
            if neighbor not in visited:
                visited[neighbor] = Node(neighbor.val, [])
                queue.append(neighbor)
                
            # Add the cloned neighbor to the current cloned node's neighbors
            visited[current].neighbors.append(visited[neighbor])
            
    return clone'''
        ]
        for i, solution in enumerate(solutions):
            ref = PlagiarismReference(
                problem_id=problem_id,
                solution_code=solution,
                solution_name=f'Reference Solution {i+1}',
                language=language
            )
            db.session.add(ref)
        db.session.commit()
    
    # Continue with the existing reference solutions
    if problem_id in reference_solutions:
        for i, solution in enumerate(reference_solutions[problem_id]):
            ref = PlagiarismReference(
                problem_id=problem_id,
                solution_code=solution,
                solution_name=f'Reference Solution {i+1}',
                language=language
            )
            db.session.add(ref)
        db.session.commit()
