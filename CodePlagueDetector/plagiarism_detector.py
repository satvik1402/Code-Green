import ast
import re
import difflib
import logging
import numpy as np
from typing import List, Dict, Any
import os

# Try to import transformers and sentence_transformers
try:
    from sentence_transformers import SentenceTransformer, util
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Semantic similarity will be disabled.")

class PlagiarismDetector:
    def __init__(self):
        self.model = None
        if TRANSFORMERS_AVAILABLE:
            try:
                # Use CodeBERT model for code similarity
                model_name = os.getenv("CODEBERT_MODEL", "microsoft/codebert-base")
                self.model = SentenceTransformer(model_name)
                logging.info(f"Loaded model: {model_name}")
            except Exception as e:
                logging.error(f"Failed to load CodeBERT model: {e}")
                self.model = None
    
    def analyze_code(self, submitted_code: str, reference_codes: List[str], problem_context: str = "", problem_id: str = "", test_passed: bool = True) -> Dict[str, Any]:
        """Analyze submitted code for plagiarism against reference solutions"""
        logging.info(f"Starting plagiarism analysis with {len(reference_codes)} reference solutions for problem {problem_id}")
        
        # Initialize feedback items list
        feedback_items = []
        
        # Store problem ID for context-aware detection
        self.current_problem_id = problem_id
        
        # Skip detailed analysis if tests failed - just do a basic check
        if not test_passed:
            logging.info("Test cases failed. Performing minimal plagiarism check.")
            
            return {
                'exact_match_detected': False,
                'structural_similarity_detected': False,
                'similarity_score': 0.0,
                'feedback': [{
                    'type': 'error',
                    'message': 'Code must pass all test cases before detailed plagiarism detection is performed.',
                    'severity': 'high'
                }],
                'test_passed': False
            }
        
        # Initialize results
        results = {
            'exact_match_detected': False,
            'variable_renaming_detected': False,
            'structural_similarity_detected': False,
            'comment_similarity_detected': False,
            'similarity_score': 0.0,
            'feedback': feedback_items,  # Use our already initialized feedback items list
            'test_passed': test_passed,
            'refactoring_detected': False
        }
        
        # If tests didn't pass, add feedback about it
        if not test_passed:
            results['feedback'].append({
                'type': 'warning',
                'message': 'Your code failed test cases for this problem. It might not be a valid solution.',
                'severity': 'medium'
            })
        
        # Clean and normalize the submitted code
        cleaned_submitted = self._clean_code(submitted_code)
        
        # Check each reference solution
        for i, reference_code in enumerate(reference_codes):
            # Try direct string comparison first (important for exact copies)
            if submitted_code.strip() == reference_code.strip():
                logging.info(f"Direct exact match detected with reference solution {i+1}")
                results['exact_match_detected'] = True
                results['variable_renaming_detected'] = True  # Implied by exact match
                results['structural_similarity_detected'] = True  # Implied by exact match
                
                # Detailed feedback with line information
                feedback_items.append(f"Exact match detected with reference solution {i+1}. The entire code matches a known solution exactly. This indicates potential plagiarism of a reference solution without any modifications. All lines in the submission are identical to the reference solution.")
                break  # No need to continue checking if we found an exact match
            
            # Clean the reference code for more detailed comparisons
            cleaned_reference = self._clean_code(reference_code)
            
            # 2. Variable Renaming Detection
            var_renaming_result = self._detect_variable_renaming(cleaned_submitted, cleaned_reference)
            if var_renaming_result['detected']:
                results['variable_renaming_detected'] = True
                results['structural_similarity_detected'] = True  # Variable renaming implies structural similarity
                
                # Enhanced feedback with specific pattern information
                feedback_items.append(f"Variable renaming pattern detected with reference solution {i+1}. {var_renaming_result['details']} This indicates that the core logic structure is similar, but variable names have been changed. The solution appears to use the same algorithm with renamed variables. This is a common technique used to disguise copied code.")
            
            # 3. Structural Similarity Detection
            structural_result = self._detect_structural_similarity(cleaned_submitted, cleaned_reference, problem_id)
            if 'detected' in structural_result and structural_result['detected'] and not results['structural_similarity_detected']:
                results['structural_similarity_detected'] = True
                
                # Detailed algorithm pattern feedback
                pattern_info = ""
                if 'pattern_name' in structural_result:
                    pattern_name = structural_result.get('pattern_name', '')
                    if pattern_name == 'two_pointer':
                        pattern_info = "The code appears to use a two-pointer technique common in this problem, with similar pointer initialization and movement patterns."
                    elif pattern_name == 'dp_approach':
                        pattern_info = "The code appears to use a dynamic programming approach similar to reference solutions, with similar array initialization and traversal patterns."
                
                feedback_items.append(f"Algorithm structure similarity detected with reference solution {i+1}: {structural_result['details']} {pattern_info} The algorithm implementation shows significant similarities in control flow and problem-solving approach, even though variable names may differ. This suggests understanding of the same core algorithm structure.")
            
            # 3. Check for comment similarity
            comment_result = self._detect_comment_similarity(submitted_code, reference_code)
            if comment_result['similarity'] > 0.7:
                results['comment_similarity_detected'] = True
                # Extract actual comments for more detailed feedback
                submitted_comments = self._extract_comments(submitted_code)
                reference_comments = self._extract_comments(reference_code)
                comments_sample = ", ".join(submitted_comments[:2]) if submitted_comments else "No comments found"
                
                feedback_items.append(f"Comment similarity detected: {comment_result['details']} Similar comments were found between your submission and reference solution {i+1}. Comments like '{comments_sample}' appear to be very similar to those in the reference solution. This is a strong indicator that the code may have been copied with little modification to the comments. Unique comments are expected in independently written solutions.")
        
        # 4. Semantic Similarity using CodeBERT (after checking all references)
        if self.model and reference_codes:
            semantic_score = self._compute_semantic_similarity(submitted_code, reference_codes)
            results['similarity_score'] = semantic_score
            
            # Don't flag as plagiarized if the code structure is different but conceptually similar
            # This follows the rule that properly refactored code shouldn't be flagged as plagiarism
            if not results['exact_match_detected'] and semantic_score > 0.8:
                # Determine if this is likely a refactored solution
                if not results['structural_similarity_detected'] and not results['variable_renaming_detected']:
                    # This is likely a refactored solution with the same approach
                    feedback_items.append(f"Semantic similarity score: {semantic_score:.2%}. While the AI model detects conceptual similarities with reference solutions, your implementation appears to be sufficiently different in structure and variable naming. This suggests you've implemented the correct algorithm approach in your own way, which is not considered plagiarism.")
                else:
                    # This is likely similar code with minor modifications
                    feedback_items.append(f"Semantic similarity score: {semantic_score:.2%}. The AI model has detected significant conceptual similarities with reference solutions. Combined with structural similarities, this suggests the solution may be derived from reference code with some modifications. Consider implementing the algorithm from scratch with your own unique approach.")
            elif semantic_score > 0.8:
                feedback_items.append(f"Semantic similarity score: {semantic_score:.2%}. The AI model has detected very high conceptual similarities with reference solutions, further confirming other plagiarism indicators. The semantic meaning of your code closely matches known solutions.")
            
            logging.info(f"Semantic similarity score: {semantic_score:.4f}")
        
        # Format the feedback items
        if feedback_items:
            results['feedback'] = feedback_items
        else:
            if results.get('refactoring_detected', False):
                results['feedback'] = ["Your code appears to be a refactored version of a reference solution, but it passes all tests and is considered original work."]
            else:
                results['feedback'] = ["No significant similarities detected. Your solution appears to be original."]
        
        # Add overall status summary
        if results['exact_match_detected']:
            results['feedback'] += "\n\nOverall: High level of plagiarism detected - this appears to be copied code."
        elif results['structural_similarity_detected'] and results['similarity_score'] > 0.8:
            results['feedback'] += "\n\nOverall: Moderate level of plagiarism detected - significant similarities found."
        elif results['similarity_score'] > 0.7:
            results['feedback'] += "\n\nOverall: Some similarities detected - may be coincidental due to problem constraints."
        
        logging.info(f"Final analysis complete. Exact match: {results['exact_match_detected']}, Structural similarity: {results['structural_similarity_detected']}, Semantic score: {results.get('similarity_score', 0):.4f}")
        return results
    
    def _clean_code(self, code: str) -> str:
        """Clean and normalize code for comparison"""
        # Remove comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
        
        # Remove extra whitespace
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def _detect_exact_match(self, code1: str, code2: str) -> Dict[str, Any]:
        """Detect exact matches in code"""
        if code1.strip() == code2.strip():
            return {'detected': True, 'details': 'Exact match detected'}
        else:
            return {'detected': False, 'details': ''}
    
    def _normalize_variables(self, code: str, variables: set) -> str:
        """Replace all variable names with a standard placeholder to detect renaming"""
        # Sort variables by length (descending) to avoid partial replacements
        sorted_vars = sorted(variables, key=len, reverse=True)
        normalized = code
        
        for i, var_name in enumerate(sorted_vars):
            # Use regex with word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(var_name) + r'\b'
            placeholder = f"VAR_{i}"
            normalized = re.sub(pattern, placeholder, normalized)
        
        # Log for debugging
        if variables:
            logging.debug(f"Normalized {len(variables)} variables in code")
            if len(normalized) > 200:
                logging.debug(f"Normalized code sample: {normalized[:200]}...")
            else:
                logging.debug(f"Normalized code: {normalized}")
        
        return normalized
    
    def _detect_variable_renaming(self, submitted_code: str, reference_code: str) -> Dict[str, Any]:
        """Detect variable renaming by analyzing code structure"""
        try:
            # Parse both codes into AST
            submitted_ast = ast.parse(submitted_code)
            reference_ast = ast.parse(reference_code)
            
            # Extract variable names and structure
            submitted_vars = self._extract_variables(submitted_ast)
            reference_vars = self._extract_variables(reference_ast)
            
            # Normalize variable names and compare structure
            submitted_normalized = self._normalize_variables(submitted_code, submitted_vars)
            reference_normalized = self._normalize_variables(reference_code, reference_vars)
            
            similarity = difflib.SequenceMatcher(None, submitted_normalized, reference_normalized).ratio()
            logging.info(f"Variable renaming similarity: {similarity:.4f}")
            
            if similarity > 0.5:  # 50% similarity after normalization
                return {
                    'detected': True,
                    'details': f'Code structure similarity: {similarity:.2%}',
                    'lines': 'multiple lines'
                }
            
        except SyntaxError:
            # If AST parsing fails, fall back to basic text comparison
            pass
        
        return {'detected': False, 'details': '', 'lines': ''}
    
    def _detect_structural_similarity(self, submitted_code: str, reference_code: str, problem_id: str = None) -> Dict[str, Any]:
        """Detect structural similarity using AST comparison and algorithm pattern recognition"""
        try:
            submitted_ast = ast.parse(submitted_code)
            reference_ast = ast.parse(reference_code)
            
            # Initialize algorithm match flag and details
            algorithm_match = False
            algorithm_match_detail = ""
            
            # SPECIAL PATTERN DETECTION (High-Level Algorithm Recognition)
            # Only apply when we know the problem context
            if hasattr(self, 'current_problem_id') and self.current_problem_id:
                problem_id = self.current_problem_id
                
                # Different algorithm patterns for different problems
                if problem_id == 'trapping_rain_water':
                    # First, check if the code actually computes trapped water
                    water_calculation_patterns = [
                        # Common water trapping calculation patterns
                        r'\b(water|trap|collected|result|ans)\s*\+=\s*(left_max|right_max|max_left|max_right|barrier)\s*-\s*\w+\[',
                        r'\b(water|trap|collected|result|ans)\s*\+=\s*(min|max)\(.*\)\s*-\s*\w+\[',
                        r'\bmin\(.*max.*\)\s*-\s*\w+\[',
                    ]
                    
                    # Check if the code contains any water calculation pattern
                    has_water_calc_submitted = any(re.search(pattern, submitted_code) for pattern in water_calculation_patterns)
                    has_water_calc_reference = any(re.search(pattern, reference_code) for pattern in water_calculation_patterns)
                    
                    # If either code doesn't calculate water, they can't be similar
                    if not has_water_calc_submitted or not has_water_calc_reference:
                        return {'detected': False, 'similarity': 0.0, 'details': 'Does not calculate trapped water', 'lines': ''}
                    
                    # Now check for specific algorithm implementation patterns
                    patterns = [
                        # Two-pointer approach specific patterns
                        {
                            'name': 'two_pointer',
                            'patterns': [
                                # Two pointers initialization and movement
                                (r'\b(left|right|start|end|begin|lo|hi)\s*=\s*(0|\w+\.length|len\(\w+\))', 
                                 r'\b(left|right|start|end|begin|lo|hi)\s*=\s*(\d+|\w+\.length|len\(\w+\))\s*-\s*1'),
                                # Pointer movement
                                (r'\b(left|begin|start)\s*\+=\s*1', r'\b(right|end)\s*-=\s*1'),
                                # Comparison driving the algorithm
                                (r'\bif\s+\w+\[\w+\]\s*[<>]\s*\w+\[\w+\]', r'\bwhile\s+\w+\s*[<>]\s*\w+'),
                            ]
                        },
                        # Dynamic programming approach patterns
                        {
                            'name': 'dp_approach',
                            'patterns': [
                                # Array initialization for left_max and right_max
                                (r'\b(left_max|right_max)\s*=\s*\[.*\]\s*\*\s*len', r'\b(left_max|right_max)\s*=\s*\[0\]\s*\*\s*len'),
                                # Forward/backward array filling
                                (r'\bfor\s+\w+\s+in\s+range\(1,\s*len', r'\bfor\s+\w+\s+in\s+range\(len.*-2,\s*-1,\s*-1\)'),
                                # Min/max calculation using arrays
                                (r'\bmin\(\w+\[\w+\],\s*\w+\[\w+\]\)', r'\bmax\(\w+\[\w+\-1\],\s*\w+\[\w+\]\)'),
                            ]
                        }
                    ]
                    
                    # Check for algorithm pattern matches
                    algorithm_matches = []
                    
                    for pattern_group in patterns:
                        group_name = pattern_group['name']
                        group_patterns = pattern_group['patterns']
                        
                        # Count matches for this pattern group
                        submitted_matches = 0
                        reference_matches = 0
                        
                        for pattern_pair in group_patterns:
                            left_pattern, right_pattern = pattern_pair
                            # Check submitted code
                            if (re.search(left_pattern, submitted_code, re.IGNORECASE) and 
                                re.search(right_pattern, submitted_code, re.IGNORECASE)):
                                submitted_matches += 1
                            # Check reference code
                            if (re.search(left_pattern, reference_code, re.IGNORECASE) and 
                                re.search(right_pattern, reference_code, re.IGNORECASE)):
                                reference_matches += 1
                        
                        # If both have enough matches for this algorithm pattern
                        if submitted_matches >= 1 and reference_matches >= 1:
                            match_percentage = min(submitted_matches, reference_matches) / len(group_patterns)
                            algorithm_matches.append({
                                'name': group_name,
                                'match_percentage': match_percentage,
                                'submitted_matches': submitted_matches,
                                'reference_matches': reference_matches
                            })
                    
                    # If any algorithm patterns matched
                    if algorithm_matches:
                        # Sort by match percentage
                        algorithm_matches.sort(key=lambda x: x['match_percentage'], reverse=True)
                        best_match = algorithm_matches[0]
                        
                        logging.info(f"Algorithm pattern match detected for {problem_id}! Type: {best_match['name']}, "
                                    f"Submitted: {best_match['submitted_matches']}, Reference: {best_match['reference_matches']}")
                        
                        algorithm_match = True
                        algorithm_match_detail = f" (Algorithm pattern match: {best_match['name']} approach)"
                
                elif problem_id == 'two_sum':
                    # Check for hash map implementation (common in two_sum)
                    hash_map_pattern = r'\b(dict|map|hash|{})\s*[=({]'
                    lookup_pattern = r'\b(in|\[)\s*\w+\s*(\]|:)'
                    
                    has_hash_map_submitted = re.search(hash_map_pattern, submitted_code, re.IGNORECASE) is not None
                    has_lookup_submitted = re.search(lookup_pattern, submitted_code, re.IGNORECASE) is not None
                    
                    has_hash_map_reference = re.search(hash_map_pattern, reference_code, re.IGNORECASE) is not None
                    has_lookup_reference = re.search(lookup_pattern, reference_code, re.IGNORECASE) is not None
                    
                    # Check if both use hash map approach
                    if has_hash_map_submitted and has_lookup_submitted and has_hash_map_reference and has_lookup_reference:
                        logging.info(f"Hash map algorithm pattern match detected for {problem_id}!")
                        algorithm_match = True
                        
                # Add more problem-specific patterns as needed for other problems
                
            # TRADITIONAL STRUCTURAL COMPARISON
            # ================================
            
            # Convert AST to structural representation
            submitted_structure = self._ast_to_structure(submitted_ast)
            reference_structure = self._ast_to_structure(reference_ast)
            
            # Compare structures
            similarity = difflib.SequenceMatcher(None, submitted_structure, reference_structure).ratio()
            logging.info(f"Base structural similarity: {similarity:.4f}")
            
            # Check control flow patterns
            try:
                # Extract just the control flow elements
                submitted_control_flow = self._extract_control_flow(submitted_ast)
                reference_control_flow = self._extract_control_flow(reference_ast)
                
                # If we have enough control flow elements to compare
                if submitted_control_flow and reference_control_flow:
                    control_flow_similarity = difflib.SequenceMatcher(None, submitted_control_flow, reference_control_flow).ratio()
                    logging.info(f"Control flow similarity: {control_flow_similarity:.4f}")
                    
                    # If control flow is very similar, boost similarity
                    if control_flow_similarity > 0.6:
                        similarity = max(similarity, control_flow_similarity * 0.9)
                        logging.info(f"Boosted structural similarity from control flow: {similarity:.4f}")
            except Exception as e:
                logging.warning(f"Error comparing control flow: {e}")
                # Continue with the regular similarity score
                
            # Boost similarity if algorithm pattern is detected for the specific problem
            if algorithm_match:
                # If we detected a specific algorithm pattern match, significantly boost the similarity
                pre_boost_similarity = similarity
                similarity = max(similarity, 0.75)  # Minimum 75% when patterns match
                logging.info(f"Boosted structural similarity due to algorithm match: {pre_boost_similarity:.4f} -> {similarity:.4f}")
                
                # Also add a special note to the details
                algorithm_match_detail = f" (Algorithm pattern match for {self.current_problem_id})"
            else:
                algorithm_match_detail = ""
                
            # Identify suspicious lines
            suspicious_lines = []
            try:
                # Find lines with control structures (if, for, while) and function calls that might be similar
                submitted_lines = submitted_code.split('\n')
                for i, line in enumerate(submitted_lines):
                    if any(keyword in line for keyword in ['if ', 'for ', 'while ', 'def ', 'return', '==']):  
                        suspicious_lines.append(str(i+1))
                
                line_info = ', '.join(suspicious_lines[:5])
                if len(suspicious_lines) > 5:
                    line_info += f" and {len(suspicious_lines)-5} more"
            except Exception as e:
                logging.warning(f"Failed to identify suspicious lines: {e}")
                # Return the result with detected lines
            # Lower threshold when algorithm pattern is detected
            threshold = 0.6 if algorithm_match else 0.7
            
            return {
                'detected': similarity > threshold,
                'similarity': similarity,
                'details': f"Structural similarity: {similarity:.2f}{algorithm_match_detail}",
                'lines': ", ".join(str(line) for line in suspicious_lines[:5]) if suspicious_lines else "All code"
            }
            
        except SyntaxError as e:
            logging.warning(f"AST parsing failed: {e}")
            return {
                'detected': False,
                'similarity': 0.0,
                'details': 'Could not parse code structure',
                'lines': ''
            }
    
    def _compute_semantic_similarity(self, submitted_code: str, reference_codes: List[str]) -> float:
        """Compute semantic similarity using CodeBERT model if available, otherwise fallback to text-based analysis"""
        try:
            # Check for common AI-generated code patterns
            ai_patterns = [
                # GPT tends to add explanatory comments
                '# Time complexity:',
                '# Space complexity:',
                '# Edge case',
                '# Initialize',
                '# Base case',
                '# This algorithm uses',
                '# A classic problem',
                '# Step 1:',
                '# We can solve',
                '# O(n) solution',
            ]
            
            # Check if submitted code has AI-generated patterns
            ai_pattern_matches = sum(1 for pattern in ai_patterns if pattern.lower() in submitted_code.lower())
            if ai_pattern_matches >= 2:  # If 2 or more AI patterns are found
                logging.info(f"AI-generated code pattern detected with {ai_pattern_matches} matches")
                return 1.0  # Consider it highly similar to AI-generated code
            
            # Clean the codes for comparison
            cleaned_submitted = self._clean_code_for_semantic_analysis(submitted_code)
            
            max_similarity = 0.0
            
            # Use CodeBERT if available
            if self.model is not None:
                try:
                    # Get embeddings using CodeBERT
                    submitted_embedding = self.model.encode(cleaned_submitted)
                    
                    for reference_code in reference_codes:
                        cleaned_reference = self._clean_code_for_semantic_analysis(reference_code)
                        reference_embedding = self.model.encode(cleaned_reference)
                        
                        # Compute cosine similarity between embeddings
                        similarity = float(np.dot(submitted_embedding, reference_embedding) / 
                                       (np.linalg.norm(submitted_embedding) * np.linalg.norm(reference_embedding)))
                        
                        max_similarity = max(max_similarity, similarity)
                        logging.info(f"CodeBERT similarity: {similarity:.4f}")
                    
                    # Boost similarity for partial matches to catch more plagiarism
                    if max_similarity > 0.4:  # If there's at least 40% similarity
                        # Apply boosting formula to make the detector more sensitive
                        max_similarity = 0.5 + (max_similarity * 0.5)  # Boost values over 0.4 to be higher
                        logging.info(f"Boosted similarity: {max_similarity:.4f}")
                    
                    return max_similarity
                except Exception as e:
                    logging.error(f"Error using CodeBERT for similarity: {e}. Falling back to text-based comparison.")
                    # Fall back to text-based comparison if CodeBERT fails
            
            # Fallback: Use text-based comparison with difflib
            logging.info("Using fallback text-based similarity comparison")
            for reference_code in reference_codes:
                cleaned_reference = self._clean_code_for_semantic_analysis(reference_code)
                similarity = difflib.SequenceMatcher(None, cleaned_submitted, cleaned_reference).ratio()
                max_similarity = max(max_similarity, similarity)
                logging.info(f"Text-based similarity: {similarity:.4f}")
            
            return max_similarity
            
        except Exception as e:
            logging.error(f"Error computing semantic similarity: {e}")
            return 0.0
    
    def _clean_code_for_semantic_analysis(self, code: str) -> str:
        """Clean code for semantic analysis by normalizing patterns but preserving code structure"""
        # Remove comments and normalize whitespace
        cleaned = self._clean_code(code)
        
        # Less aggressive normalization that preserves more structure
        # Replace variable names in assignments but keep the rest intact
        cleaned = re.sub(r'\b(\w+)\s*=\s*([^=\n]+)', r'var = \2', cleaned)
        
        # Normalize for loops but keep the iterable
        cleaned = re.sub(r'for\s+\w+\s+in\s+(\S+)', r'for var in \1', cleaned)
        
        # Normalize if statements but preserve condition structure
        cleaned = re.sub(r'if\s+([^:]*):', r'if \1:', cleaned)
        
        # Normalize function names but keep parameters
        cleaned = re.sub(r'def\s+\w+\s*(\([^\)]*\)):', r'def func\1:', cleaned)
        
        # Log a sample of the cleaned code for debugging
        if len(cleaned) > 200:
            logging.debug(f"Cleaned code sample for semantic analysis: {cleaned[:200]}...")
        else:
            logging.debug(f"Cleaned code sample for semantic analysis: {cleaned}")
        
        return cleaned
    
    def _detect_refactoring(self, submitted_code: str, reference_code: str) -> bool:
        """Detect if the submitted code is a refactored version of the reference code.
        This checks for variable renaming while maintaining the same structure.
        """
        try:
            # Parse both codes into AST
            submitted_ast = ast.parse(submitted_code)
            reference_ast = ast.parse(reference_code)
            
            # Extract control flow structure
            submitted_flow = self._extract_control_flow(submitted_ast)
            reference_flow = self._extract_control_flow(reference_ast)
            
            # High flow similarity but different variable names indicates refactoring
            flow_similarity = difflib.SequenceMatcher(None, submitted_flow, reference_flow).ratio()
            
            # Extract variable names
            submitted_vars = self._extract_variables(submitted_ast)
            reference_vars = self._extract_variables(reference_ast)
            
            # Check if variable names are different but structure is similar
            var_sets_different = submitted_vars != reference_vars
            structure_similar = flow_similarity > 0.8
            
            # Log for debugging
            if var_sets_different and structure_similar:
                logging.info(f"Refactoring detected: Flow similarity {flow_similarity:.2f}, different variable names")
                return True
                
            return False
            
        except Exception as e:
            logging.warning(f"Error in refactoring detection: {e}")
            return False
            
    def _detect_exact_match(self, submitted_code: str, reference_code: str) -> Dict[str, Any]:
        """Detect if the submitted code is a true exact match with the reference code.
        This is now very strict - only truly identical code or whitespace differences count.
        """
        # First, check for refactoring
        refactoring_detected = self._detect_refactoring(submitted_code, reference_code)
        
        # Try a direct string comparison without any cleaning
        # This is important for catching copy-paste submissions
        if submitted_code.strip() == reference_code.strip():
            logging.info("Direct string comparison exact match detected!")
            return {
                'detected': True, 
                'similarity': 1.0,
                'details': 'Exact match with reference solution',
                'lines': 'all',
                'refactoring_detected': False
            }
            
        # Clean up both codes by removing comments to ensure comment differences don't affect exact match
        try:
            # Try to parse both codes for a cleaner comparison
            submitted_ast = ast.parse(submitted_code)
            reference_ast = ast.parse(reference_code)
            
            # Direct string comparison (exact character match)
            if submitted_code == reference_code:
                logging.info("Exact match detected (character-by-character)")
                return {
                    'detected': True, 
                    'similarity': 1.0,
                    'details': 'Exact match with reference solution (character-by-character)',
                    'lines': 'all',
                    'refactoring_detected': False
                }
        except Exception as e:
            logging.warning(f"Error in code cleaning for exact match: {e}")
            
        # If direct comparison fails, try with normalized whitespace
        try:
            # Normalize both codes to remove whitespace and formatting differences
            submitted_clean = self._clean_code(submitted_code)
            reference_clean = self._clean_code(reference_code)
            
            if submitted_clean == reference_clean:
                logging.info("Normalized code exact match detected!")
                return {
                    'detected': True, 
                    'similarity': 1.0,
                    'details': 'Exact match with reference solution (normalized)',
                    'lines': 'all',
                    'refactoring_detected': False
                }
                
            # More sophisticated comparison
            # Compute string similarity to help debug
            similarity = difflib.SequenceMatcher(None, submitted_clean, reference_clean).ratio()
            if similarity > 0.99 and not refactoring_detected:  # Almost identical
                logging.info(f"High similarity ({similarity:.4f}) exact match detected!")
                return {
                    'detected': True, 
                    'similarity': similarity,
                    'details': f'Very high similarity match ({similarity:.2%})',
                    'lines': 'all',
                    'refactoring_detected': False
                }
            elif refactoring_detected:
                logging.info(f"Refactored code detected with similarity {similarity:.4f}")
                return {
                    'detected': False, 
                    'similarity': similarity,
                    'details': 'Code appears to be a refactored version of the reference solution',
                    'lines': 'all',
                    'refactoring_detected': True
                }
            else:
                logging.info(f"Codes are similar ({similarity:.4f}) but not exact matches")
                
        except Exception as e:
            logging.warning(f"Error in exact match detection: {e}")
                
        return {
            'detected': False,
            'similarity': 0.0,
            'details': '',
            'lines': '',
            'refactoring_detected': False
        }
            
    def _extract_variables(self, tree: ast.AST) -> set:
        """Extract variable names from AST"""
        variables = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                variables.add(node.id)
        
        return variables
    
    def _normalize_variables(self, code: str, variables: set) -> str:
        """Replace variable names with generic placeholders"""
        normalized = code
        var_mapping = {}
        counter = 0
        
        # Sort variables by length (longest first) to avoid partial replacements
        sorted_vars = sorted(variables, key=len, reverse=True)
        
        for var in sorted_vars:
            if var not in ['True', 'False', 'None', 'and', 'or', 'not', 'in', 'is']:
                placeholder = f'VAR{counter}'
                var_mapping[var] = placeholder
                normalized = re.sub(r'\b' + re.escape(var) + r'\b', placeholder, normalized)
                counter += 1
        
        return normalized
    
    def _detect_comment_similarity(self, submitted_code: str, reference_code: str) -> Dict[str, Any]:
        """Detect similarity in comments and documentation"""
        try:
            # Extract comments from both codes
            submitted_comments = self._extract_comments(submitted_code)
            reference_comments = self._extract_comments(reference_code)
            
            if not submitted_comments or not reference_comments:
                return {'similarity': 0.0, 'details': 'No comments to compare', 'lines': ''}
            
            # Compare comment similarity
            submitted_text = ' '.join(submitted_comments).lower()
            reference_text = ' '.join(reference_comments).lower()
            
            similarity = difflib.SequenceMatcher(None, submitted_text, reference_text).ratio()
            
            return {
                'similarity': similarity,
                'details': f'Comment similarity: {similarity:.2%}',
                'lines': 'comment lines'
            }
            
        except Exception as e:
            logging.warning(f"Comment similarity detection failed: {e}")
            return {'similarity': 0.0, 'details': 'Could not analyze comments', 'lines': ''}
    
    def _extract_comments(self, code: str) -> List[str]:
        """Extract comments from code"""
        comments = []
        lines = code.split('\n')
        
        for line in lines:
            # Single line comments
            if '#' in line:
                comment_part = line[line.find('#')+1:].strip()
                if comment_part:
                    comments.append(comment_part)
        
        # Multi-line comments (docstrings)
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef):
                    if (node.body and isinstance(node.body[0], ast.Expr) and 
                        isinstance(node.body[0].value, ast.Constant) and 
                        isinstance(node.body[0].value.value, str)):
                        comments.append(node.body[0].value.value)
        except:
            pass
        
        return comments

    def _extract_control_flow(self, tree: ast.AST) -> str:
        """Extract just the control flow elements from an AST
        This gives a higher-level view of the algorithm structure without implementation details
        """
        flow_parts = []
        visited_nodes = set()  # To avoid cycles and duplicates
        
        # Track algorithm patterns (for improved detection across implementations)
        patterns = {
            'two_pointer': 0,      # Two-pointer technique (common in array problems)
            'sliding_window': 0,   # Sliding window technique
            'binary_search': 0,    # Binary search pattern
            'dfs': 0,              # Depth-first search
            'bfs': 0,              # Breadth-first search
            'dynamic_prog': 0,     # Dynamic programming
            'greedy': 0,           # Greedy algorithm
        }
        
        def _visit_node(node):
            # Skip if already visited
            if id(node) in visited_nodes:
                return
            visited_nodes.add(id(node))
            
            # Extract control flow
            if isinstance(node, ast.FunctionDef):
                flow_parts.append(f"Function:{len(node.args.args)}")
                _visit_children(node)
            elif isinstance(node, ast.For):
                flow_parts.append("For")
                _visit_children(node)
            elif isinstance(node, ast.While):
                flow_parts.append("While")
                # Check for two-pointer pattern in while loops
                if _has_comparison_in_test(node) and _has_pointer_updates(node.body):
                    patterns['two_pointer'] += 1
                _visit_children(node)
            elif isinstance(node, ast.If):
                flow_parts.append("If")
                if hasattr(node, 'body') and node.body:
                    flow_parts.append("Then")
                    for child in node.body:
                        _visit_node(child)
                if hasattr(node, 'orelse') and node.orelse:
                    flow_parts.append("Else")
                    for child in node.orelse:
                        _visit_node(child)
            elif isinstance(node, ast.Try):
                flow_parts.append("Try")
                _visit_children(node)
                if hasattr(node, 'handlers') and node.handlers:
                    flow_parts.append("Except")
                    for handler in node.handlers:
                        _visit_node(handler)
            elif isinstance(node, ast.Return):
                flow_parts.append("Return")
                
        # Helper to detect if a node has comparison operators in its test condition
        # (Used to identify two-pointer patterns)
        def _has_comparison_in_test(node):
            if hasattr(node, 'test'):
                for subnode in ast.walk(node.test):
                    if isinstance(subnode, ast.Compare):
                        return True
            return False
            
        # Helper to detect updates to pointers (e.g., i += 1, j -= 1)
        def _has_pointer_updates(body):
            update_count = 0
            for node in body:
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.AugAssign):  # e.g., left += 1
                        update_count += 1
                        if update_count >= 1:  # At least one increment/decrement
                            return True
            return False
        
        def _visit_children(node):
            # Visit all children of a node
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST):
                            _visit_node(item)
                elif isinstance(value, ast.AST):
                    _visit_node(value)
        
        # Start the traversal from the root
        _visit_node(tree)
        
        # Add detected algorithm patterns to the flow representation
        for pattern, count in patterns.items():
            if count > 0:
                flow_parts.append(f"Pattern:{pattern}:{count}")
        
        # Join the flow elements into a string representation
        flow_str = "|".join(flow_parts)
        logging.debug(f"Control flow: {flow_str[:200]}..." if len(flow_str) > 200 else flow_str)
        return flow_str
    
    def _ast_to_structure(self, tree: ast.AST) -> str:
        """Convert AST to a structural representation"""
        structure_parts = []
        
        for node in ast.walk(tree):
            node_type = type(node).__name__
            if isinstance(node, ast.FunctionDef):
                structure_parts.append(f'FunctionDef:{len(node.args.args)}')
            elif isinstance(node, ast.For):
                structure_parts.append('ForLoop')
            elif isinstance(node, ast.While):
                structure_parts.append('WhileLoop')
            elif isinstance(node, ast.If):
                structure_parts.append('IfStatement')
            elif isinstance(node, ast.Assign):
                structure_parts.append('Assignment')
            elif isinstance(node, ast.Return):
                structure_parts.append('Return')
            elif isinstance(node, ast.BinOp):
                op_type = type(node.op).__name__
                structure_parts.append(f'BinOp:{op_type}')
            elif isinstance(node, ast.Call):
                # Try to get the function name if available
                func_name = ''
                if hasattr(node.func, 'id') and isinstance(node.func.id, str):
                    func_name = node.func.id
                structure_parts.append(f'FunctionCall:{func_name}:{len(node.args)}')
            elif isinstance(node, ast.Compare):
                op_types = [type(op).__name__ for op in node.ops]
                structure_parts.append(f'Compare:{"_".join(op_types)}')
            elif isinstance(node, ast.List) or isinstance(node, ast.Tuple):
                structure_parts.append(f'{node_type}:{len(node.elts)}')
            elif isinstance(node, ast.Dict):
                structure_parts.append(f'Dict:{len(node.keys)}')
            elif isinstance(node, ast.Try):
                structure_parts.append('TryExcept')
            elif isinstance(node, ast.With):
                structure_parts.append('WithBlock')
            elif isinstance(node, ast.ClassDef):
                structure_parts.append(f'ClassDef:{len(node.bases)}')
    
        # Add logging to see the extracted structure
        struct_str = '|'.join(structure_parts)
        logging.debug(f"AST Structure: {struct_str[:200]}..." if len(struct_str) > 200 else struct_str)
        return struct_str
