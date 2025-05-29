"""
DSA Problems database with 12 medium-level problems
"""

DSA_PROBLEMS = [
    {
        'id': 'two_sum',
        'title': 'Two Sum',
        'difficulty': 'Medium',
        'description': '''Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

Example:
Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].''',
        'function_signature': 'def two_sum(nums, target):'
    },
    {
        'id': 'merge_intervals',
        'title': 'Merge Intervals',
        'difficulty': 'Medium',
        'description': '''Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

Example:
Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].''',
        'function_signature': 'def merge(intervals):'
    },
    {
        'id': 'reverse_linked_list',
        'title': 'Reverse Linked List',
        'difficulty': 'Medium',
        'description': '''Given the head of a singly linked list, reverse the list, and return the reversed list.

Example:
Input: head = [1,2,3,4,5]
Output: [5,4,3,2,1]

Note: A ListNode class is defined as:
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next''',
        'function_signature': 'def reverse_list(head):'
    },
    {
        'id': 'detect_cycle_graph',
        'title': 'Detect Cycle in a Graph',
        'difficulty': 'Medium',
        'description': '''Given a directed graph, determine if it contains a cycle. The graph is represented as an adjacency list.

Example:
Input: graph = {0: [1], 1: [2], 2: [0]}
Output: True (cycle exists: 0 -> 1 -> 2 -> 0)

Input: graph = {0: [1], 1: [2], 2: []}
Output: False (no cycle)''',
        'function_signature': 'def has_cycle(graph):'
    },
    {
        'id': 'longest_substring_no_repeat',
        'title': 'Longest Substring Without Repeating Characters',
        'difficulty': 'Medium',
        'description': '''Given a string s, find the length of the longest substring without repeating characters.

Example:
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.

Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.''',
        'function_signature': 'def length_of_longest_substring(s):'
    },
    {
        'id': 'kth_largest_element',
        'title': 'Kth Largest Element in an Array',
        'difficulty': 'Medium',
        'description': '''Given an integer array nums and an integer k, return the kth largest element in the array.

Note that it is the kth largest element in the sorted order, not the kth distinct element.

Example:
Input: nums = [3,2,1,5,6,4], k = 2
Output: 5

Input: nums = [3,2,3,1,2,4,5,5,6], k = 4
Output: 4''',
        'function_signature': 'def find_kth_largest(nums, k):'
    },
    {
        'id': 'valid_parentheses',
        'title': 'Valid Parentheses',
        'difficulty': 'Medium',
        'description': '''Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:
1. Open brackets must be closed by the same type of brackets.
2. Open brackets must be closed in the correct order.

Example:
Input: s = "()[]{}"
Output: True

Input: s = "([)]"
Output: False''',
        'function_signature': 'def is_valid(s):'
    },
    {
        'id': 'binary_tree_level_order',
        'title': 'Binary Tree Level Order Traversal',
        'difficulty': 'Medium',
        'description': '''Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).

Example:
Input: root = [3,9,20,null,null,15,7]
Output: [[3],[9,20],[15,7]]

Note: TreeNode is defined as:
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right''',
        'function_signature': 'def level_order(root):'
    },
    {
        'id': 'word_ladder',
        'title': 'Word Ladder Problem',
        'difficulty': 'Medium',
        'description': '''A transformation sequence from word beginWord to word endWord using a dictionary wordList is a sequence of words beginWord -> s1 -> s2 -> ... -> sk such that:
- Every adjacent pair of words differs by a single letter.
- Every si for 1 <= i <= k is in wordList. Note that beginWord does not need to be in wordList.
- sk == endWord

Return the length of the shortest transformation sequence. If no such sequence exists, return 0.

Example:
Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
Output: 5
Explanation: "hit" -> "hot" -> "dot" -> "dog" -> "cog"''',
        'function_signature': 'def ladder_length(begin_word, end_word, word_list):'
    },
    {
        'id': 'trapping_rain_water',
        'title': 'Trapping Rain Water',
        'difficulty': 'Medium',
        'description': '''Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.

Example:
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The elevation map is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water are being trapped.''',
        'function_signature': 'def trap(height):'
    },
    {
        'id': 'maximum_subarray',
        'title': 'Maximum Subarray (Kadane\'s Algorithm)',
        'difficulty': 'Medium',
        'description': '''Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

Example:
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.

Input: nums = [5,4,-1,7,8]
Output: 23''',
        'function_signature': 'def max_subarray(nums):'
    },
    {
        'id': 'clone_graph',
        'title': 'Clone a Graph',
        'difficulty': 'Medium',
        'description': '''Given a reference of a node in a connected undirected graph, return a deep copy (clone) of the graph.

Each node in the graph contains a value (int) and a list (List[Node]) of its neighbors.

class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

Example:
Input: adjList = [[2,4],[1,3],[2,4],[1,3]]
Output: [[2,4],[1,3],[2,4],[1,3]]''',
        'function_signature': 'def clone_graph(node):'
    }
]
