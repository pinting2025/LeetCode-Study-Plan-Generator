import json
import random
import pandas as pd
from typing import List, Dict, Set, Tuple
from helper.alns import State
from helper.mp import LeetCodeOptimizer

class Parser(object):
    def __init__(self, csv_file: str):
        """Initialize the parser, saves the data from the CSV file into instance variables.
        
        Args:
            csv_file (str): Path to the CSV file containing LeetCode problems
        """
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        
        # Parse topics and companies
        if 'topics_list' not in self.df.columns:
            self.df['topics_list'] = self.df['related_topics'].apply(self._parse_list)
        else:
            self.df['topics_list'] = self.df['topics_list'].apply(self._parse_list)
            
        if 'companies_list' not in self.df.columns:
            self.df['companies_list'] = self.df['companies'].apply(self._parse_list)
        else:
            self.df['companies_list'] = self.df['companies_list'].apply(self._parse_list)
        
        # Count topics and companies
        self.df['topic_count'] = self.df['topics_list'].apply(len)
        self.df['company_count'] = self.df['companies_list'].apply(len)
        
        # Basic cleaning for numeric fields
        self.df['acceptance_rate'] = pd.to_numeric(self.df['acceptance_rate'], errors='coerce')
        self.df['likes'] = pd.to_numeric(self.df['likes'], errors='coerce').fillna(0)
        self.df['dislikes'] = pd.to_numeric(self.df['dislikes'], errors='coerce').fillna(0)
        
        # Calculate normalized scores
        max_likes = self.df['likes'].max()
        max_topic_count = self.df['topic_count'].max()
        max_company_count = self.df['company_count'].max()
        
        self.df['normalized_likes'] = self.df['likes'] / max_likes if max_likes > 0 else 0
        self.df['normalized_topic_count'] = self.df['topic_count'] / max_topic_count if max_topic_count > 0 else 0
        self.df['normalized_company_count'] = self.df['company_count'] / max_company_count if max_company_count > 0 else 0
        
        # Weight for FAANG questions
        self.df['is_premium'] = self.df['is_premium'].fillna(0).astype(int)
        self.df['asked_by_faang'] = self.df['asked_by_faang'].fillna(0).astype(int)
        
        # Calculate popularity score (exactly as in mp.py)
        self.df['popularity_score'] = (
            self.df['normalized_likes'] * 0.5 +
            self.df['asked_by_faang'] * 0.3 +
            self.df['normalized_topic_count'] * 0.2
        )
        
        # Extract all unique topics and companies
        self.all_topics = set()
        self.all_companies = set()
        
        for topics in self.df['topics_list']:
            self.all_topics.update(topics)
        
        for companies in self.df['companies_list']:
            self.all_companies.update(companies)
        
        print(f"Found {len(self.all_topics)} unique topics and {len(self.all_companies)} unique companies")
        
        # Define topic importance based on role (exactly as in mp.py)
        data_scientist_topics = [
            'Array', 'Hash Table', 'String', 'Math', 'Dynamic Programming', 
            'Sorting', 'Greedy', 'Database', 'Matrix'
        ]
        
        software_engineer_topics = [
            'Array', 'Hash Table', 'String', 'Linked List', 'Stack', 'Queue',
            'Tree', 'Binary Search', 'Graph', 'Heap', 'Trie', 'Recursion',
            'Dynamic Programming', 'Backtracking', 'Greedy', 'Design'
        ]
        
        self.topic_importance = {}
        for topic in self.all_topics:
            is_ds_topic = topic in data_scientist_topics
            is_se_topic = topic in software_engineer_topics
            
            self.topic_importance[topic] = {
                'Data Scientist': 1.0 if is_ds_topic else 0.3,
                'Software Engineer': 1.0 if is_se_topic else 0.5
            }
    
    def _parse_list(self, str_list):
        """Parse string representation of list into actual list.
        This matches the parse_list function in mp.py."""
        if pd.isna(str_list) or not str_list:
            return []
        
        try:
            if isinstance(str_list, str) and str_list.startswith('['):
                return eval(str_list)
            
            if isinstance(str_list, str) and ',' in str_list:
                return [t.strip() for t in str_list.split(',')]
            
            if isinstance(str_list, str):
                return [str_list.strip()]
            
            return []
        except:
            try:
                return json.loads(str_list.replace("'", "\""))
            except:
                return []

class Problem(object):
    def __init__(self, data: pd.Series):
        """Initialize a LeetCode problem.
        
        Args:
            data (pd.Series): Row from DataFrame containing problem data
        """
        self.id = data['id']
        self.title = data['title']
        self.difficulty = data['difficulty']
        self.topics = data['topics_list']
        self.companies = data['companies_list']
        self.acceptance_rate = data['acceptance_rate']
        self.estimated_time = data['estimated_time']
        self.popularity_score = data['popularity_score']
        self.is_premium = data['is_premium']
        self.asked_by_faang = data['asked_by_faang']
        
        # Add normalized fields needed for objective function
        if 'normalized_company_count' in data:
            self.normalized_company_count = data['normalized_company_count']
        else:
            # Fallback if not in the dataframe
            self.normalized_company_count = 0.0

class LeetCodeState(State):
    def __init__(self, name: str, problems: List[Problem], target_companies: List[str],
                 skill_level: int, target_role: str, study_period_days: int,
                 max_study_hours_per_day: int, weights: Dict[str, float], topic_importance: Dict[str, Dict[str, float]]):
        """Initialize the LeetCode study plan state.
        
        Args:
            name (str): Name of the instance
            problems (List[Problem]): List of available problems
            target_companies (List[str]): List of target companies
            skill_level (int): Current skill level (1-5)
            target_role (str): Target role (e.g., 'Software Engineer')
            study_period_days (int): Number of days available for study
            max_study_hours_per_day (int): Maximum study hours per day
            weights (Dict[str, float]): Weights for objective components
            topic_importance (Dict[str, Dict[str, float]]): Topic importance by role
        """
        super().__init__()
        self.name = name
        self.problems = problems
        self.target_companies = target_companies
        self.skill_level = skill_level
        self.target_role = target_role
        self.study_period_days = study_period_days
        self.max_study_hours_per_day = max_study_hours_per_day
        self.weights = weights
        self.topic_importance = topic_importance
        
        # Calculate total available minutes
        self.total_available_minutes = study_period_days * max_study_hours_per_day * 60
        
        # Initialize solution
        self.selected_problems = []
        self.covered_topics = set()
        self.total_time = 0
        
        # Efficient lookup tables
        self.problem_by_id = {p.id: p for p in problems}
        self.selected_problem_ids = set()
        
        # Difficulty distribution based on skill level
        self.difficulty_distribution = {
            1: {'Easy': 0.70, 'Medium': 0.30, 'Hard': 0.00},
            2: {'Easy': 0.50, 'Medium': 0.50, 'Hard': 0.00},
            3: {'Easy': 0.30, 'Medium': 0.70, 'Hard': 0.00},
            4: {'Easy': 0.20, 'Medium': 0.70, 'Hard': 0.10},
            5: {'Easy': 0.10, 'Medium': 0.65, 'Hard': 0.25}
        }
    
    def can_add_problem(self, problem: Problem) -> bool:
        """Check if a problem can be added to the solution.
        
        Args:
            problem (Problem): Problem to check
            
        Returns:
            bool: True if problem can be added, False otherwise
        """
        # Check if problem is already selected
        if problem.id in self.selected_problem_ids:
            return False
        
        # Check time constraint
        if self.total_time + problem.estimated_time > self.total_available_minutes:
            return False
        
        # # Check difficulty distribution
        # current_dist = self._get_current_difficulty_distribution()
        # target_dist = self.difficulty_distribution[self.skill_level]
        
        # # If adding this problem would make the distribution too far from target
        # if current_dist[problem.difficulty] + 1/len(self.problems) > target_dist[problem.difficulty] * 1.5:
        #     return False
        
        # For skill levels 1-3, no hard problems allowed
        if self.skill_level <= 3 and problem.difficulty == 'Hard':
            return False
        
        return True
    
    def add_problem(self, problem: Problem) -> bool:
        """Add a problem to the solution.
        
        Args:
            problem (Problem): Problem to add
            
        Returns:
            bool: True if problem was added, False otherwise
        """
        if not self.can_add_problem(problem):
            return False
        
        self.selected_problems.append(problem)
        self.selected_problem_ids.add(problem.id)
        self.covered_topics.update(problem.topics)
        self.total_time += problem.estimated_time
        
        return True
    
    def remove_problem(self, problem_id: int) -> bool:
        """Remove a problem from the solution.
        
        Args:
            problem_id (int): ID of problem to remove
            
        Returns:
            bool: True if problem was removed, False otherwise
        """
        if problem_id not in self.selected_problem_ids:
            return False
        
        # Find the problem object in selected_problems
        problem_to_remove = None
        for i, problem in enumerate(self.selected_problems):
            if problem.id == problem_id:
               problem_to_remove = problem
               self.selected_problems.pop(i)
               break
        
        if problem_to_remove is None:
            return False
            
        self.selected_problem_ids.remove(problem_id)
        self.total_time -= problem_to_remove.estimated_time
        
        # Update covered topics
        self.covered_topics = set()
        for p in self.selected_problems:
            self.covered_topics.update(p.topics)
        
        return True
    
    def _get_current_difficulty_distribution(self) -> Dict[str, float]:
        """Calculate current difficulty distribution.
        
        Returns:
            Dict[str, float]: Distribution of difficulties
        """
        if not self.selected_problems:
            return {'Easy': 0, 'Medium': 0, 'Hard': 0}
        
        total = len(self.selected_problems)
        counts = {'Easy': 0, 'Medium': 0, 'Hard': 0}
        
        for problem in self.selected_problems:
            counts[problem.difficulty] += 1
        
        return {k: v/total for k, v in counts.items()}
    
    def objective(self) -> float:
        """Calculate objective value of the current solution.
        
        Returns:
            float: Objective value
        """
        if not self.selected_problems:
            return float('-inf')
        
        # 1. Target company alignment
        target_company_obj = 0
        for problem in self.selected_problems:
            target_match = sum(1 for c in problem.companies if c in self.target_companies)
            if self.target_companies and len(self.target_companies) > 0:
                target_company_obj += target_match / len(self.target_companies)
        
        # 2. Topic coverage value
        topic_coverage_obj = sum(
            self.topic_importance.get(t, {}).get(self.target_role, 0.5)
            for t in self.covered_topics
        )
        
        # 3. Company count value - use normalized company count
        company_count_obj = 0
        for problem in self.selected_problems:
            company_count_obj += problem.normalized_company_count
        
        # 4. Acceptance rate value - divide by 100 as in mp.py
        acceptance_rate_obj = sum((p.acceptance_rate / 100) for p in self.selected_problems)
        
        # 5. Problem popularity
        problem_popularity_obj = sum(p.popularity_score for p in self.selected_problems)
        
        # 6. Difficulty value - new component
        difficulty_obj = 0
        for problem in self.selected_problems:
            # Assign higher scores to harder problems
            if problem.difficulty == 'Hard':
                difficulty_obj += 1.0
            elif problem.difficulty == 'Medium':
                difficulty_obj += 0.7
            else:  # Easy
                difficulty_obj += 0.1
        
        # Normalize difficulty objective by number of problems
        if self.selected_problems:
            difficulty_obj = difficulty_obj / len(self.selected_problems)
        
        # Combined objective function with modified weights
        objective = (
            self.weights['target_company'] * target_company_obj +
            self.weights['topic_coverage'] * topic_coverage_obj +
            self.weights['company_count'] * company_count_obj +
            self.weights['acceptance_rate'] * acceptance_rate_obj +
            self.weights['problem_popularity'] * problem_popularity_obj +
            # 0.25 * difficulty_obj  # Add difficulty component with significant weight
            self.weights['difficulty'] * difficulty_obj
        )
        
        return objective
    
    def random_initialize(self, seed=None):
        """Initialize a random solution.
        
        Args:
            seed (int, optional): Random seed
        """
        if seed is not None:
            random.seed(seed)
        
        # Clear current solution
        self.selected_problems = []
        self.selected_problem_ids = set()
        self.covered_topics = set()
        self.total_time = 0
        
        # Randomly select problems until time limit is reached
        available_problems = self.problems.copy()
        random.shuffle(available_problems)
        
        for problem in available_problems:
            if self.can_add_problem(problem):
                self.add_problem(problem)
    
    def copy(self):
        """Create a deep copy of the current state.
        
        Returns:
            LeetCodeState: Deep copy of current state
        """
        new_state = LeetCodeState(
            self.name,
            self.problems,
            self.target_companies,
            self.skill_level,
            self.target_role,
            self.study_period_days,
            self.max_study_hours_per_day,
            self.weights,
            self.topic_importance  # Include topic_importance here
        )
        
        new_state.selected_problems = self.selected_problems.copy()
        new_state.selected_problem_ids = self.selected_problem_ids.copy()
        new_state.covered_topics = self.covered_topics.copy()
        new_state.total_time = self.total_time
        
        return new_state

class LeetCode(object):
    def __init__(self, csv_file: str, params: Dict):
        """Initialize the LeetCode study plan problem.
        
        Args:
            csv_file (str): Path to the CSV file containing LeetCode problems
            params (Dict): Problem parameters
        """
        self.parser = Parser(csv_file)
        self.params = params
        
        # Create problems from DataFrame
        self.problems = []
        for _, row in self.parser.df.iterrows():
            problem = Problem(row)
            # Ensure normalized_company_count is set
            problem.normalized_company_count = row['normalized_company_count']
            self.problems.append(problem)
        
        # Create state
        self.state = LeetCodeState(
            name="leetcode_study_plan",
            problems=self.problems,
            target_companies=params.get('target_companies', ['Amazon', 'Google', 'Microsoft']),
            skill_level=params.get('skill_level', 3),
            target_role=params.get('target_role', 'Software Engineer'),
            study_period_days=params.get('study_period_days', 30),
            max_study_hours_per_day=params.get('max_study_hours_per_day', 2),
            weights=params.get('objective_weights', {
                'target_company': 0.30,
                'topic_coverage': 0.25,
                'company_count': 0.20,
                'acceptance_rate': 0.15,
                'problem_popularity': 0.10
            }),
            topic_importance=self.parser.topic_importance  # Pass the topic_importance
        )
        
    
    def construct_initial_solution(self, seed=None):
        """Construct initial solution using MP model.
        
        Args:
            seed (int, optional): Random seed for MP model
            
        Returns:
            LeetCodeState: Initial solution
        """
        # Create and solve MP model
        mp_model = LeetCodeOptimizer(self.parser.csv_file, self.params)
        mp_model.build_model()
        mp_results = mp_model.solve(time_limit=300)  # 5 minute time limit
        
        if mp_results is None:
            print("MP model failed to find solution, using random initialization")
            self.state.random_initialize(seed)
            return self.state
        
        # Clear current solution
        self.state.selected_problems = []
        self.state.selected_problem_ids = set()
        self.state.covered_topics = set()
        self.state.total_time = 0
        
        # Add problems from MP solution
        for _, problem_data in mp_results['selected_problems'].iterrows():
            problem_id = problem_data['id']
            # Find problem in our problem list by ID
            for problem in self.problems:
                if problem.id == problem_id:
                    if self.state.can_add_problem(problem):
                        self.state.add_problem(problem)
                    break
        
        return self.state
    
    def get_state(self) -> LeetCodeState:
        """Get the current state.
        
        Returns:
            LeetCodeState: Current state
        """
        return self.state
    
    def get_problems(self) -> List[Problem]:
        """Get all available problems.
        
        Returns:
            List[Problem]: List of all problems
        """
        return self.problems
    
    def get_problem_by_id(self, problem_id: int) -> Problem:
        """Get a problem by its ID.
        
        Args:
            problem_id (int): Problem ID
            
        Returns:
            Problem: Problem with given ID
        """
        return self.state.problem_by_id.get(problem_id)
    
    def get_selected_problems(self) -> List[Problem]:
        """Get currently selected problems.
        
        Returns:
            List[Problem]: List of selected problems
        """
        return self.state.selected_problems
    
    def get_covered_topics(self) -> Set[str]:
        """Get currently covered topics.
        
        Returns:
            Set[str]: Set of covered topics
        """
        return self.state.covered_topics
    
    def get_total_time(self) -> int:
        """Get total time used.
        
        Returns:
            int: Total time in minutes
        """
        return self.state.total_time
    
    def get_available_time(self) -> int:
        """Get available time remaining.
        
        Returns:
            int: Available time in minutes
        """
        return self.state.total_available_minutes - self.state.total_time
                
          


