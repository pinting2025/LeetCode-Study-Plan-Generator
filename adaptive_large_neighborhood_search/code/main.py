import argparse
import numpy.random as rnd
from typing import Dict, Any

from leetcode import LeetCode
from operators import *
from src.alns import ALNS
from src.alns.criteria import SimulatedAnnealing, HillClimbing
from src.helper import save_output


def main():
    parser = argparse.ArgumentParser(description='Run ALNS for LeetCode study plan')
    parser.add_argument('--data', type=str, required=True, help='Path to LeetCode problems CSV file')
    args = parser.parse_args()
    
    # Set random seed
    seed = 42
    rnd.seed(seed)
    
    # Define problem parameters
    params = {
        'target_companies': ['Amazon', 'Google', 'Microsoft'],
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
    
    # Initialize problem
    leetcode = LeetCode(args.data, params)
    
    # Get initial solution using MP model
    initial_state = leetcode.construct_initial_solution(seed)
    print(f"Initial solution objective: {initial_state.objective():.4f}")
    print(f"Initial problems selected: {len(initial_state.selected_problems)}")
    print(f"Initial topics covered: {len(initial_state.covered_topics)}")
    
    
    # Create ALNS object
    alns = ALNS(rnd.RandomState(seed))
    iterations = 1000
    
    # Add destroy operators
    alns.add_destroy_operator(destroy_topic_focused)
#     alns.add_destroy_operator(destroy_difficulty_focused)
    alns.add_destroy_operator(destroy_company_focused)
    alns.add_destroy_operator(destroy_random)
    
    # Add repair operators
    alns.add_repair_operator(topic_coverage_repair)
    alns.add_repair_operator(greedy_repair)
    alns.add_repair_operator(company_focused_repair)
    alns.add_repair_operator(acceptance_rate_repair)
    alns.add_repair_operator(popularity_repair)
    
    # Acceptance criterion
    criterion = SimulatedAnnealing(
        start_temperature=1000,  # High temperature to accept more solutions
        end_temperature=100,     # Gradually become more selective
        step=0.99               # Slower cooling to explore more
    )
    
#     criterion = HillClimbing()
    
    # Weights for operators
    weights = [100, 20, 5, 1]  # Strong focus on global best
    
    # Run ALNS
    result = alns.iterate(
        initial_state,
        weights,
        0.7,  # Quick operator adaptation
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
    
    # Save results
    results = {
        'objective': solution.objective(),
        'num_problems': len(solution.selected_problems),
        'topics_covered': len(solution.covered_topics),
        'selected_problems': [
            {
                'id': p.id,
                'title': p.title,
                'difficulty': p.difficulty,
                'topics': p.topics,
                'companies': p.companies,
                'acceptance_rate': p.acceptance_rate,
                'estimated_time': p.estimated_time
            }
            for p in solution.selected_problems
        ]
    }
    
    # Save to CSV
    import pandas as pd
    df = pd.DataFrame(results['selected_problems'])
    df.to_csv('alns_solution.csv', index=False)
    
    # Save to JSON
    import json
    with open('alns_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()