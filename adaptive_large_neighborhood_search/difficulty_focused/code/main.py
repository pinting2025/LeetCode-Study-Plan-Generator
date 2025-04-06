import argparse
import numpy.random as rnd
import pandas as pd
import json

from leetcode import LeetCode
from operators import *
from helper.alns import ALNS
from helper.alns.criteria import SimulatedAnnealing, HillClimbing
from helper.mp import *

def main():
    # Set random seed
    seed = 42
    rnd.seed(seed)
    
    # Define problem parameters
    params = {
        'target_companies': ['Amazon', 'Google', 'Microsoft', 'Facebook', 'Apple'],
        'skill_level': 3,  
        'target_role': 'Software Engineer', 
        'study_period_days': 30,
        'max_study_hours_per_day': 2,
        'objective_weights': {
            'target_company': 0.30,      # Reduced from 0.30
            'topic_coverage': 0.25,      # Reduced from 0.25
            'company_count': 0.20,       # Reduced from 0.20
            'acceptance_rate': 0.15,     # Kept same
            'problem_popularity': 0.10,  # Kept same
            'difficulty': 0.30      
        }
    }
    
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
    
    # Add destroy operators
    alns.add_destroy_operator(destroy_difficulty_focused)
    # alns.add_destroy_operator(destroy_topic_focused)
    # alns.add_destroy_operator(destroy_company_focused)
    alns.add_destroy_operator(destroy_random)
    
    # Add repair operators
    alns.add_repair_operator(difficulty_repair)  
    # alns.add_repair_operator(topic_coverage_repair)
    alns.add_repair_operator(greedy_repair)
    # alns.add_repair_operator(company_focused_repair)
    # alns.add_repair_operator(acceptance_rate_repair)
    # alns.add_repair_operator(popularity_repair)
    
    criterion = SimulatedAnnealing(
        start_temperature=2000,  # Higher temperature to accept more solutions initially
        end_temperature=0.1,     # Lower end temperature to be more selective later
        step=0.95               # Slower cooling to explore more
    )
    
    # criterion = HillClimbing()
    
    # Weights for operators
    weights = [150, 100, 50, 20]  # Strong focus on global best
    
    # Run ALNS
    result = alns.iterate(
        initial_state,
        weights,
        0.8,  # Quick operator adaptation
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
        study_plan.to_csv('./adaptive_large_neighborhood_search/difficulty_focused/result/alns_study_plan.csv', index=False)
        
        print("\nStudy plan created!")
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