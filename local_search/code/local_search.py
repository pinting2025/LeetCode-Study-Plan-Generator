import pandas as pd
import numpy as np
import networkx as nx
import argparse
from collections import defaultdict
import ast
import random

def parse_list(text):
    """
    Parse a string representation of a list into an actual list.
    Example: "Array,Hash Table" -> ["Array", "Hash Table"]
    """
    if isinstance(text, list):
        return text
    try:
        if isinstance(text, str):
            if text.startswith('[') and text.endswith(']'):
                return ast.literal_eval(text)
            elif ',' in text:
                return [item.strip() for item in text.split(',') if item.strip()]
        return []
    except:
        return []

# Create a replacement for the LeetCodeOptimizer class
class LeetCodeOptimizer:
    """
    Simplified LeetCodeOptimizer class that only contains the necessary TOPIC_PREREQS constant.
    """
    # Define topic prerequisites
    TOPIC_PREREQS = {
        "Array": [],
        "String": ["Array"],
        "Linked List": ["Array"],
        "Stack": ["Array", "Linked List"],
        "Queue": ["Array", "Linked List"],
        "Hash Table": ["Array"],
        "Tree": ["Linked List"],
        "Heap": ["Array", "Tree"],
        "Graph": ["Array", "Tree"],
        "Trie": ["Tree", "Hash Table"], 
        "Math": [],
        "Bit Manipulation": ["Math"],
        "Two Pointers": ["Array"],
        "Sliding Window": ["Two Pointers", "Array"],
        "Binary Search": ["Array"],
        "Recursion": [],
        "Backtracking": ["Recursion"],
        "Greedy": ["Sort"],
        "Sort": ["Array", "Math"],
        "Design": [],
        "Binary Indexed Tree": ["Array", "Binary Search", "Bit Manipulation"],
        "Segment Tree": ["Tree", "Binary Search", "Recursion"],
        "Union Find": ["Array", "Tree"],
        "Ordered Map": ["Hash Table"],
        "Dynamic Programming": ["Recursion", "Array"],
        "Divide and Conquer": ["Recursion"],
        "Depth-first Search": ["Graph", "Tree", "Recursion", "Stack"],
        "Breadth-first Search": ["Graph", "Tree", "Queue"],
        "Topological Sort": ["Graph", "Depth-first Search"],
        "Line Sweep": ["Sort"],
        "Brainteaser": ["Math"],         
    }
    # Can add more constants if needed

class LocalSearchOptimizer:
    """
    Local Search Optimizer for LeetCode study plans.
    This class adjusts an existing study plan when a user is facing difficulties.
    """
    
    def __init__(self, study_plan_path, leetcode_data_path='leetcode_v2.csv', target_companies=None):
        """
        Initialize the local search optimizer.
        
        Parameters:
        -----------
        study_plan_path : str
            Path to the CSV containing the existing study plan
        leetcode_data_path : str
            Path to the LeetCode dataset
        target_companies : list
            List of target companies (default: ['Amazon', 'Google', 'Microsoft'])
        """
        self.study_plan_path = study_plan_path
        self.leetcode_data_path = leetcode_data_path
        self.target_companies = target_companies or ['Amazon', 'Google', 'Microsoft']
        
        # Load study plan
        self.study_plan = pd.read_csv(study_plan_path)
        
        # Load LeetCode data
        self.leetcode_data = pd.read_csv(leetcode_data_path)
        
        # Check which column names exist in the study plan
        study_plan_columns = self.study_plan.columns.tolist()
        
        # Map the possible column names to standardized names
        column_mappings = {
            'topics': ['topics', 'topics_list', 'related_topics'],
            'companies': ['companies', 'companies_list']
        }
        
        # Determine which columns to use
        topics_col = next((col for col in column_mappings['topics'] if col in study_plan_columns), None)
        companies_col = next((col for col in column_mappings['companies'] if col in study_plan_columns), None)
        
        if topics_col:
            # Create a standardized 'topics' column
            self.study_plan['topics'] = self.study_plan[topics_col].apply(self._safe_eval)
        else:
            print("Warning: No topics column found in study plan")
            self.study_plan['topics'] = [[]] * len(self.study_plan)
        
        if companies_col:
            # Create a standardized 'companies' column
            self.study_plan['companies'] = self.study_plan[companies_col].apply(self._safe_eval)
        else:
            print("Warning: No companies column found in study plan")
            self.study_plan['companies'] = [[]] * len(self.study_plan)
        
        self.leetcode_data['topics_list'] = self.leetcode_data['related_topics'].apply(parse_list)
        self.leetcode_data['companies_list'] = self.leetcode_data['companies'].apply(parse_list)
        
        # Import topic prerequisites from the LeetCodeOptimizer class
        self.topic_prereqs = LeetCodeOptimizer.TOPIC_PREREQS
        
        # Define difficulty scores
        self.difficulty_scores = {
            'Easy': 15,
            'Medium': 25,
            'Hard': 40
        }
        
        # Define objective function weights
        self.weights = {
            'difficulty_trend': 0.25,   # Smooth progression in difficulty
            'topic_coverage': 0.20,     # Coverage of prerequisite topics
            'acceptance_rate': 0.20,    # Higher acceptance rates (≥ 50%)
            'target_company': 0.15,     # Coverage of problems from target companies
            'company_count': 0.10,      # Number of companies that ask these problems
            'problem_popularity': 0.10  # Based on acceptance rate as proxy
        }
        
        # Build prerequisite graph
        self.prereq_graph = self._build_prereq_graph()
    
    def _safe_eval(self, val):
        """Safely evaluate string representations of lists"""
        if isinstance(val, list):
            return val
        try:
            if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
                return ast.literal_eval(val)
            return []
        except:
            return []
    
    def _build_prereq_graph(self):
        """Build a directed graph of topic prerequisites"""
        G = nx.DiGraph()
        
        # Get all topics from the study plan
        all_topics = set()
        for topics in self.study_plan['topics']:
            if isinstance(topics, list):
                all_topics.update(topics)
        
        # Add all topics as nodes
        for topic in all_topics:
            G.add_node(topic)
        
        # Add edges based on prerequisites
        for topic, prereqs in self.topic_prereqs.items():
            if topic in all_topics:
                for prereq in prereqs:
                    if prereq in all_topics:
                        G.add_edge(prereq, topic)
        
        return G
    
    def identify_critical_topics(self, day):
        """
        Identify topics with the most prerequisites from the given day's plan
        
        Parameters:
        -----------
        day : int
            The day in the study plan to analyze
            
        Returns:
        --------
        list
            List of (topic, prereq_count, frequency) tuples sorted by prereq count
        """
        # Check if 'day' column exists in the study plan
        if 'day' not in self.study_plan.columns:
            print(f"Error: No 'day' column found in study plan. Available columns: {', '.join(self.study_plan.columns)}")
            return []
            
        # Get problems for the given day
        today_plan = self.study_plan[self.study_plan['day'] == day]
        
        if today_plan.empty:
            print(f"No problems found for day {day}")
            return []
        
        # Collect all topics from today's problems
        topic_frequency = defaultdict(int)
        for _, problem in today_plan.iterrows():
            topics = problem['topics']
            if isinstance(topics, list):
                for topic in topics:
                    topic_frequency[topic] += 1
        
        # Count prerequisites for each topic
        topic_prereq_count = {}
        for topic in topic_frequency.keys():
            try:
                # Get all prerequisites using NetworkX ancestors function
                prereqs = list(nx.ancestors(self.prereq_graph, topic))
                topic_prereq_count[topic] = len(prereqs)
            except:
                topic_prereq_count[topic] = 0
        
        # Sort topics by prerequisite count (most prereqs first)
        sorted_topics = sorted(
            [(topic, prereq_count, topic_frequency[topic]) 
             for topic, prereq_count in topic_prereq_count.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_topics
    
    def get_all_prerequisites(self, topics):
        """
        Get all prerequisites for given topics
        
        Parameters:
        -----------
        topics : list
            List of topic names
            
        Returns:
        --------
        set
            Set of all prerequisite topics
        """
        all_prereqs = set()
        for topic in topics:
            try:
                # Get all ancestors (prerequisites) using NetworkX
                prereqs = nx.ancestors(self.prereq_graph, topic)
                all_prereqs.update(prereqs)
            except:
                continue
        
        # Include the topics themselves
        all_prereqs.update(topics)
        
        return all_prereqs
    
    def calculate_difficulty_trend(self, problems):
        """
        Calculate a score for the difficulty trend of a sequence of problems
        
        Parameters:
        -----------
        problems : DataFrame
            DataFrame containing problem information
            
        Returns:
        --------
        float
            Score representing how well the difficulty progresses (higher is better)
        """
        if len(problems) <= 1:
            return 0
        
        trend_score = 0
        problems_sorted = problems.sort_values('day')
        
        prev_difficulty = None
        for _, problem in problems_sorted.iterrows():
            curr_difficulty = self.difficulty_scores.get(problem['difficulty'], 0)
            
            if prev_difficulty is not None:
                diff = curr_difficulty - prev_difficulty
                
                # Reward gradual increases (0 < diff < 15)
                if 0 < diff < 15:
                    trend_score += 1
                # Penalize decreases or large jumps
                elif diff < 0:
                    trend_score -= 0.5
                elif diff >= 15:
                    trend_score -= 0.25
            
            prev_difficulty = curr_difficulty
        
        # Normalize by the number of transitions
        if len(problems) > 1:
            trend_score = trend_score / (len(problems) - 1)
        
        return trend_score
    
    def calculate_topic_coverage_score(self, problems, critical_topics):
        """
        Calculate how well the problems cover critical topics and their prerequisites
        
        Parameters:
        -----------
        problems : DataFrame
            DataFrame containing problem information
        critical_topics : set
            Set of critical topics that need coverage
            
        Returns:
        --------
        float
            Score representing topic coverage quality
        """
        if not critical_topics:
            return 0
        
        # Get all topics covered by the problems
        covered_topics = set()
        for _, problem in problems.iterrows():
            if isinstance(problem['topics'], list):
                covered_topics.update(problem['topics'])
        
        # Get all prerequisites for critical topics
        prereqs_needed = self.get_all_prerequisites(critical_topics)
        
        # Calculate coverage of prerequisites
        prereqs_covered = len(prereqs_needed.intersection(covered_topics))
        prereqs_coverage_ratio = prereqs_covered / len(prereqs_needed) if prereqs_needed else 0
        
        return prereqs_coverage_ratio
    
    def calculate_acceptance_rate_score(self, problems):
        """
        Calculate a score based on acceptance rates
        
        Parameters:
        -----------
        problems : DataFrame
            DataFrame containing problem information
            
        Returns:
        --------
        float
            Score representing average acceptance rate (normalized)
        """
        if problems.empty:
            return 0
            
        # Calculate average acceptance rate, with preference for rates ≥ 50%
        rates = problems['acceptance_rate'].values
        avg_rate = np.mean(rates) / 100  # Normalize to 0-1 scale
        
        # Bonus for having most problems with acceptance rate ≥ 50%
        high_rate_count = sum(1 for rate in rates if rate >= 50)
        high_rate_ratio = high_rate_count / len(rates) if len(rates) > 0 else 0
        
        # Combined score (weighted average)
        return 0.7 * avg_rate + 0.3 * high_rate_ratio
    
    def calculate_target_company_score(self, problems):
        """
        Calculate how well problems cover target companies
        
        Parameters:
        -----------
        problems : DataFrame
            DataFrame containing problem information
            
        Returns:
        --------
        float
            Score representing target company coverage
        """
        if problems.empty or not self.target_companies:
            return 0
        
        # Count problems by target company
        company_coverage = defaultdict(int)
        
        for _, problem in problems.iterrows():
            companies = problem['companies']
            if isinstance(companies, list):
                for company in self.target_companies:
                    if company in companies:
                        company_coverage[company] += 1
        
        # Calculate average coverage across target companies
        avg_coverage = np.mean([company_coverage[company] for company in self.target_companies])
        
        # Normalize by the number of problems
        max_possible = len(problems)
        normalized_coverage = min(avg_coverage / max_possible, 1.0) if max_possible > 0 else 0
        
        return normalized_coverage
    
    def calculate_company_count_score(self, problems):
        """
        Calculate a score based on the number of companies asking each problem
        
        Parameters:
        -----------
        problems : DataFrame
            DataFrame containing problem information
            
        Returns:
        --------
        float
            Score representing company count (normalized)
        """
        if problems.empty:
            return 0
        
        # Calculate average company count per problem
        company_counts = [len(companies) if isinstance(companies, list) else 0 
                         for companies in problems['companies']]
        
        avg_count = np.mean(company_counts)
        
        # Normalize by a reasonable maximum (e.g., 20 companies per problem)
        max_companies = 20
        normalized_count = min(avg_count / max_companies, 1.0)
        
        return normalized_count
    
    def calculate_problem_popularity_score(self, problems):
        """
        Calculate a score based on problem popularity (using acceptance rate as a proxy)
        
        Parameters:
        -----------
        problems : DataFrame
            DataFrame containing problem information
            
        Returns:
        --------
        float
            Score representing problem popularity
        """
        if problems.empty:
            return 0
        
        # Use acceptance rate as a proxy for popularity
        # Higher acceptance rate correlates with clearer problem statement and better user experience
        # Additionally, we can use problem ID as a proxy for popularity (lower IDs are older and more popular)
        
        # Normalize problem IDs (older problems tend to be more popular)
        ids = problems['id'].values
        max_id = self.leetcode_data['id'].max()
        normalized_ids = [1 - (id / max_id) for id in ids]  # Lower ID = higher score
        
        # Combine with acceptance rates
        acceptance_rates = problems['acceptance_rate'].values / 100  # Normalize to 0-1
        
        # Average the scores (70% weight to acceptance rate, 30% to ID-based score)
        popularity_scores = [0.7 * acc + 0.3 * id_score 
                            for acc, id_score in zip(acceptance_rates, normalized_ids)]
        
        return np.mean(popularity_scores)
    
    def evaluate_plan(self, plan, critical_topics):
        """
        Evaluate a study plan using the multi-objective function
        
        Parameters:
        -----------
        plan : DataFrame
            Study plan to evaluate
        critical_topics : set
            Set of critical topics that need coverage
            
        Returns:
        --------
        dict
            Dictionary of component scores and total score
        """
        # Calculate component scores
        difficulty_trend_score = self.calculate_difficulty_trend(plan)
        topic_coverage_score = self.calculate_topic_coverage_score(plan, critical_topics)
        acceptance_rate_score = self.calculate_acceptance_rate_score(plan)
        target_company_score = self.calculate_target_company_score(plan)
        company_count_score = self.calculate_company_count_score(plan)
        problem_popularity_score = self.calculate_problem_popularity_score(plan)
        
        # Combine scores based on weights
        total_score = (
            self.weights['difficulty_trend'] * difficulty_trend_score +
            self.weights['topic_coverage'] * topic_coverage_score +
            self.weights['acceptance_rate'] * acceptance_rate_score +
            self.weights['target_company'] * target_company_score +
            self.weights['company_count'] * company_count_score +
            self.weights['problem_popularity'] * problem_popularity_score
        )
        
        return {
            'total': total_score,
            'difficulty_trend': difficulty_trend_score,
            'topic_coverage': topic_coverage_score,
            'acceptance_rate': acceptance_rate_score,
            'target_company': target_company_score,
            'company_count': company_count_score,
            'problem_popularity': problem_popularity_score
        }
    
    def generate_neighborhood(self, current_plan, day, planning_days, critical_topics):
        """
        Generate neighboring solutions by making small changes to the current plan
        
        Parameters:
        -----------
        current_plan : DataFrame
            Current study plan
        day : int
            Starting day for modifications
        planning_days : int
            Number of days to consider for modifications
        critical_topics : set
            Set of critical topics to focus on
            
        Returns:
        --------
        list
            List of neighbor plans (DataFrames)
        """
        neighbors = []
        
        # Define the day range to consider
        day_range = range(day, day + planning_days)
        plan_subset = current_plan[current_plan['day'].isin(day_range)].copy()
        
        # Get all problem IDs in the current plan to avoid duplicates
        existing_problem_ids = set(current_plan['id'].values)
        
        # Get candidate problems that match our critical topics
        candidate_problems = self.leetcode_data[
            (self.leetcode_data['topics_list'].apply(
                lambda topics: any(topic in critical_topics for topic in topics)
            )) &
            (self.leetcode_data['acceptance_rate'] >= 50) &
            (~self.leetcode_data['id'].isin(existing_problem_ids))
        ].copy()
        
        if candidate_problems.empty:
            print("Warning: No suitable candidate problems found")
            return neighbors
        
        # Neighbor generation strategies:
        
        # 1. Replace a problem with a new one from candidates
        for i, row in plan_subset.iterrows():
            for _, candidate in candidate_problems.sample(min(3, len(candidate_problems))).iterrows():
                new_plan = current_plan.copy()
                
                # Create a new row for the candidate problem
                new_row = {
                    'day': row['day'],
                    'id': candidate['id'],
                    'title': candidate['title'],
                    'difficulty': candidate['difficulty'],
                    'topics': candidate['topics_list'],
                    'companies': candidate['companies_list'],
                    'estimated_time': candidate.get('estimated_time', self.difficulty_scores.get(candidate['difficulty'], 30)),
                    'acceptance_rate': candidate['acceptance_rate']
                }
                
                # Replace the problem
                new_plan.loc[i] = new_row
                
                # Add to neighbors
                neighbors.append(new_plan)
        
        # 2. Swap problems between days
        days_to_swap = list(day_range)
        if len(days_to_swap) > 1:
            for _ in range(min(5, len(plan_subset))):
                day1, day2 = random.sample(days_to_swap, 2)
                
                new_plan = current_plan.copy()
                day1_problems = new_plan[new_plan['day'] == day1]
                day2_problems = new_plan[new_plan['day'] == day2]
                
                if not day1_problems.empty and not day2_problems.empty:
                    # Select a random problem from each day
                    prob1_idx = random.choice(day1_problems.index)
                    prob2_idx = random.choice(day2_problems.index)
                    
                    # Swap the days
                    temp_day = new_plan.loc[prob1_idx, 'day']
                    new_plan.loc[prob1_idx, 'day'] = new_plan.loc[prob2_idx, 'day']
                    new_plan.loc[prob2_idx, 'day'] = temp_day
                    
                    neighbors.append(new_plan)
        
        # 3. Add a new problem (if there's room in the day's time budget)
        max_time_per_day = 120  # 2 hours per day (120 minutes)
        
        for day_num in day_range:
            day_problems = current_plan[current_plan['day'] == day_num]
            day_time = day_problems['estimated_time'].sum()
            
            if day_time < max_time_per_day:
                time_left = max_time_per_day - day_time
                
                # Filter candidates that fit in the remaining time
                fitting_candidates = candidate_problems[
                    candidate_problems['estimated_time'] <= time_left
                ]
                
                if not fitting_candidates.empty:
                    for _, candidate in fitting_candidates.sample(min(3, len(fitting_candidates))).iterrows():
                        new_plan = current_plan.copy()
                        
                        # Create a new row for the candidate problem
                        new_row = pd.DataFrame([{
                            'day': day_num,
                            'id': candidate['id'],
                            'title': candidate['title'],
                            'difficulty': candidate['difficulty'],
                            'topics': candidate['topics_list'],
                            'companies': candidate['companies_list'],
                            'estimated_time': candidate.get('estimated_time', self.difficulty_scores.get(candidate['difficulty'], 30)),
                            'acceptance_rate': candidate['acceptance_rate']
                        }])
                        
                        # Add the new problem
                        new_plan = pd.concat([new_plan, new_row], ignore_index=True)
                        
                        neighbors.append(new_plan)
        
        # 4. Remove a problem (to make room for better ones in future iterations)
        if len(plan_subset) > 3:  # Only if we have enough problems to remove one
            for _ in range(min(3, len(plan_subset))):
                new_plan = current_plan.copy()
                idx_to_remove = random.choice(plan_subset.index)
                new_plan = new_plan.drop(idx_to_remove).reset_index(drop=True)
                neighbors.append(new_plan)
        
        return neighbors
    
    def perform_local_search(self, trigger_day, planning_days=7, iterations=50):
        """
        Perform local search to adjust the study plan
        
        Parameters:
        -----------
        trigger_day : int
            The day at which the user is struggling
        planning_days : int
            Number of days to plan ahead (default: 7)
        iterations : int
            Maximum number of iterations for the local search
            
        Returns:
        --------
        DataFrame
            The modified study plan
        """
        print(f"Performing local search from day {trigger_day} for {planning_days} days")
        
        # 1. Identify critical topics from the trigger day
        critical_topics_info = self.identify_critical_topics(trigger_day)
        
        if not critical_topics_info:
            print(f"No critical topics found for day {trigger_day}")
            return self.study_plan
        
        print("\nCritical topics (topic, prereq count, frequency):")
        for topic_info in critical_topics_info[:5]:
            print(f"- {topic_info[0]}: {topic_info[1]} prerequisites, appears {topic_info[2]} times")
        
        # 2. Get top 3 critical topics and all their prerequisites
        top_topics = [t[0] for t in critical_topics_info[:3]]
        critical_topics = set(self.get_all_prerequisites(top_topics))
        
        print(f"\nFound {len(critical_topics)} prerequisite topics to focus on:")
        print(", ".join(sorted(critical_topics)))
        
        # 3. Determine planning horizon
        max_day = self.study_plan['day'].max()
        remaining_days = max_day - trigger_day + 1
        adjusted_planning_days = min(remaining_days, planning_days)
        
        if adjusted_planning_days <= 0:
            print("No remaining days to plan")
            return self.study_plan
        
        # 4. Extract the relevant segment of the study plan for local search
        day_range = range(trigger_day, trigger_day + adjusted_planning_days)
        plan_segment = self.study_plan[self.study_plan['day'].isin(day_range)].copy()
        
        # Initialize the current best plan and its score
        current_plan = self.study_plan.copy()
        current_score = self.evaluate_plan(plan_segment, critical_topics)['total']
        
        print(f"\nInitial plan evaluation score: {current_score:.4f}")
        
        # Local search iterations
        improvement_found = True
        iteration = 0
        
        while improvement_found and iteration < iterations:
            improvement_found = False
            iteration += 1
            
            # Generate neighbors
            neighbors = self.generate_neighborhood(current_plan, trigger_day, adjusted_planning_days, critical_topics)
            
            if not neighbors:
                print(f"No valid neighbors generated at iteration {iteration}")
                break
                
            print(f"Iteration {iteration}: Generated {len(neighbors)} neighboring plans")
            
            # Evaluate neighbors
            best_neighbor = None
            best_neighbor_score = current_score
            
            for neighbor in neighbors:
                neighbor_segment = neighbor[neighbor['day'].isin(day_range)]
                neighbor_score = self.evaluate_plan(neighbor_segment, critical_topics)['total']
                
                if neighbor_score > best_neighbor_score:
                    best_neighbor = neighbor
                    best_neighbor_score = neighbor_score
            
            # Update current plan if improvement found
            if best_neighbor is not None and best_neighbor_score > current_score:
                current_plan = best_neighbor
                current_score = best_neighbor_score
                improvement_found = True
                print(f"  Found improvement: new score = {current_score:.4f}")
            else:
                print("  No improvement found in this iteration")
        
        print(f"\nLocal search completed after {iteration} iterations")
        print(f"Final plan score: {current_score:.4f}")
        
        # Calculate final detailed scores
        final_segment = current_plan[current_plan['day'].isin(day_range)]
        final_scores = self.evaluate_plan(final_segment, critical_topics)
        
        print("\nDetailed scores:")
        for component, score in final_scores.items():
            if component != 'total':
                print(f"- {component}: {score:.4f}")
        
        # Ensure the modified plan is properly sorted
        modified_plan = current_plan.sort_values('day').reset_index(drop=True)
        
        return modified_plan
    
    def save_modified_plan(self, modified_plan, output_path=None):
        """
        Save the modified study plan to a CSV file
        
        Parameters:
        -----------
        modified_plan : DataFrame
            The modified study plan
        output_path : str, optional
            Path to save the modified plan (default: 'modified_study_plan.csv')
        """
        if output_path is None:
            output_path = 'modified_study_plan.csv'
        
        # Convert lists to strings for CSV output
        modified_plan_copy = modified_plan.copy()
        for col in ['topics', 'companies']:
            if col in modified_plan_copy.columns:
                modified_plan_copy[col] = modified_plan_copy[col].apply(lambda x: str(x) if isinstance(x, list) else x)
        
        modified_plan_copy.to_csv(output_path, index=False)
        print(f"\nModified study plan saved to {output_path}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Local search for LeetCode study plan adjustment')
    parser.add_argument('--study_plan', type=str, default='mathematical_programming/result/study_plan.csv', 
                        help='Path to the existing study plan CSV')
    parser.add_argument('--leetcode_data', type=str, default='data\_leetcode_v2.csv',
                        help='Path to the LeetCode dataset CSV')
    parser.add_argument('--day', type=int, required=True,
                        help='Day at which the user is struggling')
    parser.add_argument('--planning_days', type=int, default=7,
                        help='Number of days to plan ahead')
    parser.add_argument('--iterations', type=int, default=50,
                        help='Maximum number of iterations for local search')
    parser.add_argument('--output', type=str, default='modified_study_plan.csv',
                        help='Path to save the modified plan')
    parser.add_argument('--target_companies', type=str, default='Amazon,Google,Microsoft',
                        help='Comma-separated list of target companies')
    
    args = parser.parse_args()
    
    # Parse target companies
    target_companies = [company.strip() for company in args.target_companies.split(',')]
    
    # Create the local search optimizer
    optimizer = LocalSearchOptimizer(
        args.study_plan, 
        args.leetcode_data,
        target_companies=target_companies
    )
    
    # Identify critical topics
    critical_topics_info = optimizer.identify_critical_topics(args.day)
    if not critical_topics_info:
        print(f"No critical topics found for day {args.day}")
        return
        
    # Get top 3 critical topics and all their prerequisites
    top_topics = [t[0] for t in critical_topics_info[:3]]
    critical_topics = set(optimizer.get_all_prerequisites(top_topics))
    
    # Determine planning horizon
    max_day = optimizer.study_plan['day'].max()
    remaining_days = max_day - args.day + 1
    adjusted_planning_days = min(remaining_days, args.planning_days)
    
    # Evaluate the original study plan
    day_range = range(args.day, args.day + adjusted_planning_days)
    original_segment = optimizer.study_plan[optimizer.study_plan['day'].isin(day_range)].copy()
    original_scores = optimizer.evaluate_plan(original_segment, critical_topics)
    
    print("\nOriginal Study Plan Evaluation:")
    print(f"Total score: {original_scores['total']:.4f}")
    print("\nDetailed scores:")
    for component, score in original_scores.items():
        if component != 'total':
            print(f"- {component}: {score:.4f}")
    
    # Perform local search
    modified_plan = optimizer.perform_local_search(
        args.day, 
        args.planning_days,
        args.iterations
    )
    
    # Save the modified plan
    optimizer.save_modified_plan(modified_plan, args.output)
    
    print(f"\nLocal search completed. Modified study plan saved to {args.output}")
if __name__ == "__main__":
    main()