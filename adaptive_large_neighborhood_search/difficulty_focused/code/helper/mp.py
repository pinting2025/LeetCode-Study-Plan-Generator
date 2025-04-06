import pandas as pd
import numpy as np
import json
from docplex.mp.model import Model
from collections import defaultdict
import time
import networkx as nx
import cplex

def parse_list(str_list):
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

def normalize(values, min_val=None, max_val=None):
    if min_val is None:
        min_val = min(values)
    if max_val is None:
        max_val = max(values)
    
    if max_val == min_val:
        return [1.0] * len(values)
    
    return [(v - min_val) / (max_val - min_val) for v in values]

class LeetCodeOptimizer:
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
    
    SKILL_LEVEL_FULFILLED_PREREQS = {
        1: ["Array", "Math", "Recursion"],  # Beginner
        2: ["Array", "Math", "Recursion", "String", "Hash Table", "Two Pointers", "Binary Search", "Sort"],  # Basic
        3: ["Array", "Math", "Recursion", "String", "Hash Table", "Two Pointers", "Binary Search", 
            "Sort", "Linked List", "Stack", "Queue", "Bit Manipulation", "Greedy"],  # Intermediate
        4: ["Array", "Math", "Recursion", "String", "Hash Table", "Two Pointers", "Binary Search", 
            "Sort", "Linked List", "Stack", "Queue", "Bit Manipulation", "Greedy", 
            "Tree", "Graph", "Dynamic Programming", "Backtracking", "Divide and Conquer"],  # Advanced
        5: [], 
    }
    
    def __init__(self, data_path, params):
        self.data_path = data_path
        self.params = params
        
        # Default parameters if not provided
        self.target_companies = params.get('target_companies', ['Amazon', 'Google', 'Microsoft'])
        self.skill_level = params.get('skill_level', 3)
        self.target_role = params.get('target_role', 'Software Engineer')
        self.study_period_days = params.get('study_period_days', 30)
        self.max_study_hours_per_day = params.get('max_study_hours_per_day', 2)
        
        # Objective weights
        self.weights = params.get('objective_weights', {
            'target_company': 0.25,      # Reduced from 0.30
            'topic_coverage': 0.15,      # Reduced from 0.25
            'company_count': 0.10,       # Reduced from 0.20
            'acceptance_rate': 0.10,     # Kept same
            'problem_popularity': 0.10,  # Kept same
            'difficulty': 0.30 
        })
    
        
        # Calculate total available minutes
        self.total_available_minutes = self.study_period_days * self.max_study_hours_per_day * 60
        
        # Define difficulty distribution based on skill level
        self.difficulty_distribution = {
            1: {'Easy': 0.70, 'Medium': 0.30, 'Hard': 0.00},
            2: {'Easy': 0.50, 'Medium': 0.50, 'Hard': 0.00},
            3: {'Easy': 0.30, 'Medium': 0.70, 'Hard': 0.00},
            4: {'Easy': 0.20, 'Medium': 0.70, 'Hard': 0.10},
            5: {'Easy': 0.10, 'Medium': 0.65, 'Hard': 0.25}
        }
        
        # Load and process data
        self.df = None
        self.all_topics = set()
        self.all_companies = set()
        self.topic_importance = {}
        self.process_data()
        
        # Build prerequisite graph
        self.prereq_graph = self._build_prereq_graph()
        
        # Get fulfilled prerequisites based on skill level
        self.fulfilled_prereqs = self._get_fulfilled_prereqs()
        
        # Build the model
        self.model = None
        self.selected_vars = None
        self.topic_covered_vars = None
        self.results = None
        
    def process_data(self):
        print("Processing LeetCode data...")
        df = pd.read_csv(self.data_path)
        
        # Basic cleaning for numeric fields
        df['acceptance_rate'] = pd.to_numeric(df['acceptance_rate'], errors='coerce')
        df['likes'] = pd.to_numeric(df['likes'], errors='coerce').fillna(0)
        df['dislikes'] = pd.to_numeric(df['dislikes'], errors='coerce').fillna(0)
        
        # Parse topics and companies
        if 'topics_list' not in df.columns:
            df['topics_list'] = df['related_topics'].apply(parse_list)
        else:
            df['topics_list'] = df['topics_list'].apply(parse_list)
            
        if 'companies_list' not in df.columns:
            df['companies_list'] = df['companies'].apply(parse_list)
        else:
            df['companies_list'] = df['companies_list'].apply(parse_list)
        
        # Count topics and companies
        df['topic_count'] = df['topics_list'].apply(len)
        df['company_count'] = df['companies_list'].apply(len)
        
        # Difficulty scores
        difficulty_map = {'Easy': 1, 'Medium': 2, 'Hard': 3}
        df['difficulty_score'] = df['difficulty'].map(difficulty_map)
        
        # Estimated time if not already in the dataset
        if 'estimated_time' not in df.columns:
            time_map = {'Easy': 30, 'Medium': 60, 'Hard': 120}  # minutes
            df['estimated_time'] = df['difficulty'].map(time_map)
        
        # Calculate popularity score based on likes, submissions, etc.
        max_likes = df['likes'].max()
        max_topic_count = df['topic_count'].max()
        max_company_count = df['company_count'].max()
        
        df['normalized_likes'] = df['likes'] / max_likes if max_likes > 0 else 0
        df['normalized_topic_count'] = df['topic_count'] / max_topic_count if max_topic_count > 0 else 0
        df['normalized_company_count'] = df['company_count'] / max_company_count if max_company_count > 0 else 0
        
        # Weight for FAANG questions
        df['is_premium'] = df['is_premium'].fillna(0).astype(int)
        df['asked_by_faang'] = df['asked_by_faang'].fillna(0).astype(int)
        
        # Calculate popularity score
        df['popularity_score'] = (
            df['normalized_likes'] * 0.5 +
            df['asked_by_faang'] * 0.3 +
            df['normalized_topic_count'] * 0.2
        )
        
        # Extract all unique topics and companies
        all_topics = set()
        all_companies = set()
        
        for topics in df['topics_list']:
            all_topics.update(topics)
        
        for companies in df['companies_list']:
            all_companies.update(companies)
        
        print(f"Found {len(all_topics)} unique topics and {len(all_companies)} unique companies")
        
        # Define topic importance based on role
        data_scientist_topics = [
            'Array', 'Hash Table', 'String', 'Math', 'Dynamic Programming', 
            'Sorting', 'Greedy', 'Database', 'Matrix'
        ]
        
        software_engineer_topics = [
            'Array', 'Hash Table', 'String', 'Linked List', 'Stack', 'Queue',
            'Tree', 'Binary Search', 'Graph', 'Heap', 'Trie', 'Recursion',
            'Dynamic Programming', 'Backtracking', 'Greedy', 'Design'
        ]
        
        topic_importance = {}
        for topic in all_topics:
            is_ds_topic = topic in data_scientist_topics
            is_se_topic = topic in software_engineer_topics
            
            topic_importance[topic] = {
                'Data Scientist': 1.0 if is_ds_topic else 0.3,
                'Software Engineer': 1.0 if is_se_topic else 0.5
            }
        
        # Store processed data
        self.df = df
        self.all_topics = all_topics
        self.all_companies = all_companies
        self.topic_importance = topic_importance
        
    def _build_prereq_graph(self):
        """Build a directed graph of topic prerequisites"""
        G = nx.DiGraph()
        
        # Add all topics from our dataset
        for topic in self.all_topics:
            G.add_node(topic)
        
        # Add edges based on prerequisites
        for topic, prereqs in self.TOPIC_PREREQS.items():
            if topic in self.all_topics:
                for prereq in prereqs:
                    if prereq in self.all_topics:
                        G.add_edge(prereq, topic)
        
        return G
    
    def _get_fulfilled_prereqs(self):
        if self.skill_level in self.SKILL_LEVEL_FULFILLED_PREREQS:
            return set(self.SKILL_LEVEL_FULFILLED_PREREQS[self.skill_level])
        return set()
    
    def build_model(self):
        print("Building the optimization model...")
        self.model = Model(name="LeetCode_Study_Plan_Optimization")
        
        # 1. Binary variable for selecting a problem
        self.selected_vars = self.model.binary_var_dict(self.df.index, name="selected")
        
        # 2. Binary variable for topic coverage
        self.topic_covered_vars = self.model.binary_var_dict(self.all_topics, name="topic_covered")
        
        # 3. Binary variable for tracking prerequisite fulfillment
        self.topic_prereq_fulfilled_vars = self.model.binary_var_dict(self.all_topics, name="prereq_fulfilled")
        
        # Constraints
        # 1. Time constraint
        self.model.add_constraint(
            self.model.sum(self.selected_vars[i] * self.df.iloc[i]['estimated_time'] for i in self.df.index) <= self.total_available_minutes,
            ctname="total_time_constraint"
        )
        
        # 2. Topic coverage constraints
        for topic in self.all_topics:
            # A topic is covered if at least one problem covering it is selected
            problems_with_topic = [i for i in self.df.index if topic in self.df.iloc[i]['topics_list']]
            
            if problems_with_topic:
                self.model.add_constraint(
                    self.topic_covered_vars[topic] <= self.model.sum(self.selected_vars[i] for i in problems_with_topic),
                    ctname=f"topic_coverage_{topic}"
                )
        
        # 3. Prerequisite constraints
        # Set pre-fulfilled topics
        for topic in self.fulfilled_prereqs:
            if topic in self.all_topics:
                self.model.add_constraint(
                    self.topic_prereq_fulfilled_vars[topic] == 1,
                    ctname=f"prefulfilled_{topic}"
                )
        
        # For other topics, check prerequisites
        for topic in self.all_topics:
            if topic not in self.fulfilled_prereqs:
                # Get immediate prerequisites for this topic
                prereqs = list(self.prereq_graph.predecessors(topic))
                
                if prereqs:
                    # A topic's prerequisites are fulfilled if all immediate prerequisites are covered
                    self.model.add_constraint(
                        self.topic_prereq_fulfilled_vars[topic] <= self.model.sum(
                            self.topic_prereq_fulfilled_vars[p] for p in prereqs
                        ) / len(prereqs),
                        ctname=f"prereq_fulfilled_{topic}"
                    )
                else:
                    # If no prerequisites, it's automatically fulfilled
                    self.model.add_constraint(
                        self.topic_prereq_fulfilled_vars[topic] == 1,
                        ctname=f"no_prereq_{topic}"
                    )
                    
        # 4. A topic can only be covered if its prerequisites are fulfilled
        for topic in self.all_topics:
            self.model.add_constraint(
                self.topic_covered_vars[topic] <= self.topic_prereq_fulfilled_vars[topic],
                ctname=f"cover_if_prereq_{topic}"
            )
        
        # 5. Difficulty distribution constraints based on skill level
        current_dist = self.difficulty_distribution[self.skill_level]
        
        # Get indices for each difficulty
        easy_indices = self.df[self.df['difficulty'] == 'Easy'].index
        medium_indices = self.df[self.df['difficulty'] == 'Medium'].index
        hard_indices = self.df[self.df['difficulty'] == 'Hard'].index
        
        # Total selected problems
        total_selected = self.model.sum(self.selected_vars[i] for i in self.df.index)
        
        # Difficulty distribution constraints with some flexibility
        if current_dist['Easy'] > 0:
            self.model.add_constraint(
                self.model.sum(self.selected_vars[i] for i in easy_indices) >= 
                (current_dist['Easy'] * total_selected) - 2,
                ctname="min_easy_problems"
            )
            
            self.model.add_constraint(
                self.model.sum(self.selected_vars[i] for i in easy_indices) <= 
                (current_dist['Easy'] * total_selected) + 2,
                ctname="max_easy_problems"
            )
        
        if current_dist['Medium'] > 0:
            self.model.add_constraint(
                self.model.sum(self.selected_vars[i] for i in medium_indices) >= 
                (current_dist['Medium'] * total_selected) - 2,
                ctname="min_medium_problems"
            )
            
            self.model.add_constraint(
                self.model.sum(self.selected_vars[i] for i in medium_indices) <= 
                (current_dist['Medium'] * total_selected) + 2,
                ctname="max_medium_problems"
            )
        
        if current_dist['Hard'] > 0:
            self.model.add_constraint(
                self.model.sum(self.selected_vars[i] for i in hard_indices) >= 
                (current_dist['Hard'] * total_selected) - 2,
                ctname="min_hard_problems"
            )
            
            self.model.add_constraint(
                self.model.sum(self.selected_vars[i] for i in hard_indices) <= 
                (current_dist['Hard'] * total_selected) + 2,
                ctname="max_hard_problems"
            )
        
        # Handle skill level 1-3: No hard problems
        if self.skill_level <= 3:
            for i in hard_indices:
                self.model.add_constraint(self.selected_vars[i] == 0, ctname=f"no_hard_problems_{i}")
        
        # 6. Minimum topic coverage
        self.model.add_constraint(
            self.model.sum(self.topic_covered_vars[t] for t in self.all_topics) >= 5,  # Cover at least 5 topics
            ctname="min_topic_coverage"
        )
        
        # # 7. Maximum number of problems constraint
        # max_problems = min(50, self.study_period_days * 2)  # At most 2 problems per day
        # self.model.add_constraint(
        #     total_selected <= max_problems,
        #     ctname="max_problems"
        # )
        
        # # 8. Minimum number of problems constraint
        # min_problems = max(10, self.study_period_days // 3)  # At least 1 problem every 3 days
        # self.model.add_constraint(
        #     total_selected >= min_problems,
        #     ctname="min_problems"
        # )
        
        # 9. Difficulty progression within topics
        # For each topic, ensure easy problems come before medium, and medium before hard
        for topic in self.all_topics:
            # Get indices of problems for this topic by difficulty
            topic_easy = [i for i in self.df.index if 
                         topic in self.df.iloc[i]['topics_list'] and 
                         self.df.iloc[i]['difficulty'] == 'Easy']
            
            topic_medium = [i for i in self.df.index if 
                           topic in self.df.iloc[i]['topics_list'] and 
                           self.df.iloc[i]['difficulty'] == 'Medium']
            
            topic_hard = [i for i in self.df.index if 
                         topic in self.df.iloc[i]['topics_list'] and 
                         self.df.iloc[i]['difficulty'] == 'Hard']
            
            # If no problems of a certain difficulty, skip this constraint
            if not topic_easy or not topic_medium:
                continue
                
            # If any medium problem is selected, at least one easy problem must be selected
            if topic_medium:
                self.model.add_constraint(
                    self.model.sum(self.selected_vars[i] for i in topic_medium) <= 
                    self.model.sum(self.selected_vars[i] for i in topic_easy) * len(topic_medium),
                    ctname=f"easy_before_medium_{topic}"
                )
            
            # If any hard problem is selected, at least one medium problem must be selected
            if topic_hard and topic_medium:
                self.model.add_constraint(
                    self.model.sum(self.selected_vars[i] for i in topic_hard) <= 
                    self.model.sum(self.selected_vars[i] for i in topic_medium) * len(topic_hard),
                    ctname=f"medium_before_hard_{topic}"
                )
        
        # Objective function components
        # 1. Target company alignment
        target_company_obj = 0
        for i in self.df.index:
            companies = self.df.iloc[i]['companies_list']
            target_match = sum(1 for c in companies if c in self.target_companies)
            if self.target_companies and len(self.target_companies) > 0:
                target_company_obj += self.selected_vars[i] * (target_match / len(self.target_companies))
        
        # 2. Topic coverage value
        topic_coverage_obj = self.model.sum(
            self.topic_covered_vars[t] * self.topic_importance.get(t, {}).get(self.target_role, 0.5) 
            for t in self.all_topics
        )
        
        # 3. Company count value
        company_count_obj = self.model.sum(
            self.selected_vars[i] * self.df.iloc[i]['normalized_company_count'] 
            for i in self.df.index
        )
        
        # 4. Acceptance rate value
        acceptance_rate_obj = self.model.sum(
            self.selected_vars[i] * (self.df.iloc[i]['acceptance_rate'] / 100) 
            for i in self.df.index
        )
        
        # 5. Problem popularity
        problem_popularity_obj = self.model.sum(
            self.selected_vars[i] * self.df.iloc[i]['popularity_score'] 
            for i in self.df.index
        )
        
        # 6. Difficulty
        difficulty_obj = self.model.sum(
    self.selected_vars[i] * (
        1.0 if self.df.iloc[i]['difficulty'] == 'Hard' else
        0.5 if self.df.iloc[i]['difficulty'] == 'Medium' else
        -1.0  # Easy problems
    ) for i in self.df.index
)

        # Combined objective function
        objective = (
            self.weights['target_company'] * target_company_obj +
            self.weights['topic_coverage'] * topic_coverage_obj +
            self.weights['company_count'] * company_count_obj +
            self.weights['acceptance_rate'] * acceptance_rate_obj +
            self.weights['problem_popularity'] * problem_popularity_obj + 
            self.weights['difficulty'] * difficulty_obj
        )

        # Set the objective
        self.model.maximize(objective)
        
        return self.model
    
    def solve(self, time_limit=None):
        """
        Solve the model and return the results
        
        Parameters:
        -----------
        time_limit : int, optional
            Time limit for the solver in seconds
        
        Returns:
        --------
        results : dict
            Dictionary containing the optimization results
        """
        if self.model is None:
            self.build_model()
        
        print("Solving the optimization model...")
        start_time = time.time()
        
        # Set time limit if provided
        if time_limit is not None:
            self.model.set_time_limit(time_limit)
        
        # Solve the model
        solution = self.model.solve()
        
        end_time = time.time()
        solution_time = end_time - start_time
        
        if not solution:
            print("No solution found")
            return None
        
        print(f"Objective value: {solution.get_objective_value()}")
        print(f"Solution time: {solution_time:.2f} seconds")
        
        # Extract the selected problems
        selected_problems = []
        for i in self.df.index:
            if self.selected_vars[i].solution_value > 0.5:  # Binary variable is selected
                problem = self.df.iloc[i].copy()
                selected_problems.append(problem)
        
        # Convert to DataFrame
        selected_df = pd.DataFrame(selected_problems)
        
        # Get covered topics
        covered_topics = [t for t in self.all_topics 
                         if t in self.topic_covered_vars and 
                         self.topic_covered_vars[t].solution_value > 0.5]
        
        # Calculate statistics
        difficulty_counts = selected_df['difficulty'].value_counts().to_dict()
        
        # Difficulty distribution percentages
        total_problems = len(selected_problems)
        difficulty_percentages = {
            diff: (count / total_problems * 100) if total_problems > 0 else 0 
            for diff, count in difficulty_counts.items()
        }
        
        print("\nSelected Problem Statistics:")
        print(f"Total problems: {total_problems}")
        print(f"Difficulty distribution: {difficulty_counts}")
        print(f"Difficulty percentages: {difficulty_percentages}")
        print(f"Topics covered: {len(covered_topics)} out of {len(self.all_topics)}")
        
        # Calculate target company coverage
        company_coverage = {}
        for company in self.target_companies:
            company_problems = [
                p for _, p in selected_df.iterrows() 
                if company in p['companies_list']
            ]
            company_coverage[company] = len(company_problems)
        
        print("\nTarget Company Coverage:")
        for company, count in company_coverage.items():
            print(f"{company}: {count} problems")
        
        # Calculate topic coverage
        topic_coverage = {}
        for topic in covered_topics:
            topic_problems = [
                p for _, p in selected_df.iterrows() 
                if topic in p['topics_list']
            ]
            topic_coverage[topic] = len(topic_problems)
        
        print("\nTop Topics Coverage:")
        for topic, count in sorted(topic_coverage.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"{topic}: {count} problems")
        
        # Store the results
        self.results = {
            'objective_value': solution.get_objective_value(),
            'solution_time': solution_time,
            'selected_problems': selected_df,
            'difficulty_distribution': difficulty_counts,
            'difficulty_percentages': difficulty_percentages,
            'covered_topics': covered_topics,
            'company_coverage': company_coverage,
            'topic_coverage': topic_coverage
        }
        
        return self.results
    
    def create_study_plan(self):
        """
        Create a daily study plan based on the optimization results
        
        Returns:
        --------
        study_plan : pd.DataFrame
            Daily study plan
        """
        if self.results is None:
            print("No optimization results available. Run solve() first.")
            return None
        
        selected_df = self.results['selected_problems']
        
        # Implement a topological sort of topics based on prerequisites
        topic_order = {}
        sorted_topics = list(nx.topological_sort(self.prereq_graph))
        for i, topic in enumerate(sorted_topics):
            if topic in self.results['covered_topics']:
                topic_order[topic] = i
        
        # Function to calculate the priority score for a problem
        def problem_priority(problem):
            # Get the minimum order of topics in this problem
            problem_topics = [t for t in problem['topics_list'] if t in topic_order]
            min_topic_order = min([topic_order.get(t, float('inf')) for t in problem_topics]) if problem_topics else float('inf')
            
            # Priority: first by topic order, then by difficulty, then by acceptance rate
            return (min_topic_order, problem['difficulty_score'], -problem['acceptance_rate'])
        
        # Sort problems by priority
        sorted_problems = selected_df.copy()
        sorted_problems['priority'] = sorted_problems.apply(problem_priority, axis=1)
        sorted_problems = sorted_problems.sort_values('priority')
        
        # Create a daily plan
        daily_plan = []
        day = 1
        day_minutes = 0
        max_minutes_per_day = self.max_study_hours_per_day * 60
        
        for _, problem in sorted_problems.iterrows():
            # If adding this problem exceeds the daily limit, move to next day
            if day_minutes + problem['estimated_time'] > max_minutes_per_day:
                day += 1
                day_minutes = 0
            
            # Add problem to the plan
            daily_plan.append({
                'day': day,
                'id': problem['id'],
                'title': problem['title'],
                'difficulty': problem['difficulty'],
                'topics': problem['topics_list'],
                'companies': problem['companies_list'],
                'estimated_time': problem['estimated_time'],
                'acceptance_rate': problem['acceptance_rate']
            })
            
            day_minutes += problem['estimated_time']
        
        # Convert to DataFrame
        plan_df = pd.DataFrame(daily_plan)
        
        # Calculate cumulative study time
        plan_df['cumulative_time'] = plan_df.groupby('day')['estimated_time'].cumsum()
        
        # Make sure we don't exceed the study period
        plan_df = plan_df[plan_df['day'] <= self.study_period_days]
        
        return plan_df

def main():
    # Define parameters
    params = {
        'target_companies': ['Amazon', 'Google', 'Microsoft', 'Facebook', 'Apple'],
        'skill_level': 3,  
        'target_role': 'Software Engineer', 
        'study_period_days': 30,
        'max_study_hours_per_day': 2,
        'objective_weights': {
            'target_company': 0.30,
            'topic_coverage': 0.25,
            'company_count': 0.20,
            'acceptance_rate': 0.15,
            'problem_popularity': 0.10
        }
    }
    
    # Create optimizer
    optimizer = LeetCodeOptimizer('data/_leetcode_v2.csv', params)
    
    # Build and solve the model
    optimizer.build_model()
    results = optimizer.solve(time_limit=300)  # 5 minute time limit
    
    if results:
        # Create study plan
        study_plan = optimizer.create_study_plan()
        
        # Save results
        results['selected_problems'].to_csv('mathematical_programming/result/selected_problems.csv', index=False)
        study_plan.to_csv('mathematical_programming/result/study_plan.csv', index=False)
        
        print("\nStudy plan created!")
        print(f"Selected {len(results['selected_problems'])} problems over {params['study_period_days']} days")
        print("Results saved to 'selected_problems.csv' and 'study_plan.csv'")
        
        # Print sample of the study plan
        print("\nSample of your study plan (first 5 days):")
        days_sample = study_plan[study_plan['day'] <= 5]
        for day, group in days_sample.groupby('day'):
            total_time = group['estimated_time'].sum()
            print(f"\nDay {day} (Total: {total_time} minutes):")
            for _, problem in group.iterrows():
                print(f"  - {problem['title']} ({problem['difficulty']}, {problem['estimated_time']} min)")
                print(f"    Topics: {', '.join(problem['topics'][:3])}" + 
                     (f" +{len(problem['topics'])-3} more" if len(problem['topics']) > 3 else ""))

if __name__ == "__main__":
    main()