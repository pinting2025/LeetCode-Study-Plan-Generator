import argparse
import numpy.random as rnd
import pandas as pd
import json

from leetcode import LeetCode
from operators import *
from helper.alns import ALNS
from helper.alns.criteria import SimulatedAnnealing, HillClimbing
from helper.mp import *

def get_valid_input(prompt, valid_options=None, input_type=int, min_val=None, max_val=None):
    """Get valid user input with error handling.
    
    Args:
        prompt (str): Input prompt
        valid_options (list, optional): List of valid options for string input
        input_type (type): Type to convert input to (int or str)
        min_val (int, optional): Minimum value for numeric input
        max_val (int, optional): Maximum value for numeric input
        
    Returns:
        Validated user input
    """
    while True:
        try:
            user_input = input(prompt)
            if input_type == int:
                value = int(user_input)
                if min_val is not None and value < min_val:
                    print(f"Please enter a number >= {min_val}")
                    continue
                if max_val is not None and value > max_val:
                    print(f"Please enter a number <= {max_val}")
                    continue
                return value
            else:
                if valid_options and user_input not in valid_options:
                    print(f"Please enter one of: {', '.join(valid_options)}")
                    continue
                return user_input
        except ValueError:
            print("Invalid input. Please try again.")

def main():
    # Set random seed
    seed = 42
    rnd.seed(seed)
    
    print("\nWelcome to the LeetCode Study Plan Generator!")
    print("Please provide the following information to create your personalized study plan.\n")
    
    # Get target companies
    print("Available companies: Amazon, Google, Microsoft, Facebook, Apple, and more")
    print("Enter target companies (comma-separated):")
    companies_input = input().strip()
    target_companies = [company.strip() for company in companies_input.split(',')]
    
    # Get skill level
    print("\nSkill Level Guide:")
    print("1: Beginner")
    print("2: Basic")
    print("3: Intermediate")
    print("4: Advanced")
    print("5: Expert")
    skill_level = get_valid_input("Enter your skill level (1-5): ", input_type=int, min_val=1, max_val=5)
    
    # Get target role
    valid_roles = ['Software Engineer', 'Data Scientist']
    print("\nAvailable roles:", ', '.join(valid_roles))
    target_role = get_valid_input("Enter your target role: ", valid_options=valid_roles, input_type=str)
    
    # Get study period
    study_period_days = get_valid_input(
        "\nEnter study period in days (1-90): ",
        input_type=int,
        min_val=1,
        max_val=90
    )
    
    # Get study hours per day
    max_study_hours_per_day = get_valid_input(
        "\nEnter maximum study hours per day (1-8): ",
        input_type=int,
        min_val=1,
        max_val=8
    )
    
    # Define problem parameters
    params = {
        'target_companies': target_companies,
        'skill_level': skill_level,
        'target_role': target_role,
        'study_period_days': study_period_days,
        'max_study_hours_per_day': max_study_hours_per_day,
        'objective_weights': {
            'target_company': 0.25,      # Reduced from 0.30
            'topic_coverage': 0.15,      # Reduced from 0.25
            'company_count': 0.10,       # Reduced from 0.20
            'acceptance_rate': 0.10,     # Kept same
            'problem_popularity': 0.10,  # Kept same
            'difficulty': 0.30           # New weight for difficulty
        }
    }
    
    print("\nGenerating your personalized study plan...")
    
    # Initialize problem
    data = './data/_leetcode_v2.csv'
    leetcode = LeetCode(data, params)
    
    # Get initial solution using MP model
    initial_state = leetcode.construct_initial_solution(seed)
    print(f"Initial solution objective: {initial_state.objective():.4f}")
    print(f"Initial problems selected: {len(initial_state.selected_problems)}")
    print(f"Initial topics covered: {len(initial_state.covered_topics)}")
    
    # Create ALNS object
    alns = ALNS(rnd.RandomState(seed))
    iterations = 1000
    
    # Add destroy operators with modified weights to focus on difficulty
    alns.add_destroy_operator(destroy_difficulty_focused)  # Prioritize difficulty-focused destroy
    alns.add_destroy_operator(destroy_topic_focused)      # Keep topic coverage
    alns.add_destroy_operator(destroy_company_focused)    # Keep company coverage
    alns.add_destroy_operator(destroy_random)             # Keep some randomness
    
    # Add repair operators with modified weights
    alns.add_repair_operator(greedy_repair)              # Prioritize objective value
    alns.add_repair_operator(difficulty_repair)          # Focus on difficulty
    alns.add_repair_operator(topic_coverage_repair)      # Keep topic coverage
    alns.add_repair_operator(company_focused_repair)     # Keep company coverage
    alns.add_repair_operator(acceptance_rate_repair)     # Keep some easier problems for balance
    alns.add_repair_operator(popularity_repair)          # Keep popular problems
    
    # Use Simulated Annealing instead of Hill Climbing to allow more exploration
    criterion = SimulatedAnnealing(
        start_temperature=2000,  # Higher temperature to accept more solutions initially
        end_temperature=0.1,     # Lower end temperature to be more selective later
        step=0.95               # Slower cooling to explore more
    )
    
    # Modified weights to focus on difficulty
    weights = [150, 100, 50, 20]  # Strong focus on difficulty and topic coverage
    
    # Run ALNS with modified parameters
    result = alns.iterate(
        initial_state,
        weights,
        0.8,  # Faster operator adaptation
        criterion,
        iterations=iterations,
        collect_stats=True
    )
    
    # Get final solution
    if result:
        solution = result.best_state
        print(f"\nALNS complete - Objective: {solution.objective():.4f}")
        print(f"Problems selected: {len(solution.selected_problems)}")
        print(f"Topics covered: {len(solution.covered_topics)}")
    else:
        print("\nNo improvement found. Using initial solution.")
        solution = initial_state
    
    print("\nFinal Results:")
    print(f"Best objective value: {solution.objective():.4f}")
    print(f"Number of problems selected: {len(solution.selected_problems)}")
    print(f"Topics covered: {len(solution.covered_topics)}")

    # create study plan
    if result:
        solution = result.best_state
        optimizer = LeetCodeOptimizer(data, params)
        
        # Transfer the ALNS solution to the optimizer format
        selected_problems_df = pd.DataFrame([
            {
                'id': p.id,
                'title': p.title,
                'difficulty': p.difficulty,
                'topics_list': p.topics,
                'companies_list': p.companies,
                'acceptance_rate': p.acceptance_rate,
                'estimated_time': p.estimated_time,
                'difficulty_score': 1 if p.difficulty == 'Easy' else 2 if p.difficulty == 'Medium' else 3
            }
            for p in solution.selected_problems
        ])
    
        # Store the results in the optimizer format
        optimizer.results = {
            'selected_problems': selected_problems_df,
            'covered_topics': list(solution.covered_topics)
        }
        
        # Now you can call create_study_plan
        study_plan = optimizer.create_study_plan()
        
        # Save results
        study_plan.to_csv('./adaptive_large_neighborhood_search/result/alns_study_plan.csv', index=False)
        
        print("\nStudy plan created!")
        print("Results saved to 'alns_study_plan.csv'")
        
        # # Print sample of the study plan
        # print("\nSample of your study plan (first 5 days):")
        # days_sample = study_plan[study_plan['day'] <= 5]
        # for day, group in days_sample.groupby('day'):
        #     total_time = group['estimated_time'].sum()
        #     print(f"\nDay {day} (Total: {total_time} minutes):")
        #     for _, problem in group.iterrows():
        #         print(f"  - {problem['title']} ({problem['difficulty']}, {problem['estimated_time']} min)")
        #         print(f"    Topics: {', '.join(problem['topics'][:3])}" + 
        #              (f" +{len(problem['topics'])-3} more" if len(problem['topics']) > 3 else ""))

if __name__ == "__main__":
    main()