from leetcode import LeetCodeState, Problem

### Destroy operators ###

def destroy_random(current: LeetCodeState, random_state, destroy_factor=0.3):
    """Random destroy operator
    
    Randomly removes problems from the solution
    
    Args:
        current: LeetCodeState
            Current state before destroying
        random_state: numpy.random.RandomState
            Random state for reproducibility
        destroy_factor: float
            Percentage of problems to remove
    Returns:
        destroyed: LeetCodeState
            State after destroying
    """
    destroyed = current.copy()
    
    # Get all selected problems
    selected_problems = destroyed.selected_problems.copy()
    
    if not selected_problems:
        return destroyed
    
    # Determine number of problems to remove
    num_to_remove = max(1, int(len(selected_problems) * destroy_factor))
    num_to_remove = min(num_to_remove, len(selected_problems))
    
    # Randomly select problems to remove
    indices = random_state.choice(len(selected_problems), num_to_remove, replace=False)
    
    # Remove selected problems
    for idx in sorted(indices, reverse=True):
        problem = selected_problems[idx]
        destroyed.remove_problem(problem.id)
    
    return destroyed

def destroy_topic_focused(current: LeetCodeState, random_state, destroy_factor=0.3):
    """Topic-focused destroy operator
    
    Removes problems that cover specific topics to allow for better topic coverage
    
    Args:
        current: LeetCodeState
            Current state before destroying
        random_state: random_state
            Random state for reproducibility
        destroy_factor: float
            Percentage of problems to remove
    Returns:
        destroyed: LeetCodeState
            State after destroying
    """
    destroyed = current.copy()
    
    if not destroyed.selected_problems:
        return destroyed
    
    # Determine number of problems to remove
    num_to_remove = max(1, int(len(destroyed.selected_problems) * destroy_factor))
    num_to_remove = min(num_to_remove, len(destroyed.selected_problems))
    
    # Select a random topic from currently covered topics
    covered_topics = list(destroyed.covered_topics)
    if not covered_topics:
        return destroyed
    
    target_topic = random_state.choice(covered_topics)
    
    # Find problems covering the target topic
    topic_problems = [
        problem for problem in destroyed.selected_problems
        if target_topic in problem.topics
    ]
    
    # Randomly select problems to remove
    if topic_problems:
        num_to_remove = min(num_to_remove, len(topic_problems))
        indices = random_state.choice(len(topic_problems), num_to_remove, replace=False)
        
        # Store problem IDs to remove
        problem_ids_to_remove = []
        for idx in indices:
            problem = topic_problems[idx]
            problem_ids_to_remove.append(problem.id)
        
        # Remove problems by ID (more reliable than by object)
        for problem_id in problem_ids_to_remove:
            destroyed.remove_problem(problem_id)
    
    return destroyed

def destroy_difficulty_focused(current: LeetCodeState, random_state, destroy_factor=0.3):
    """Difficulty-focused destroy operator
    
    Removes problems of a specific difficulty to improve difficulty distribution,
    with a bias towards removing easier problems.
    
    Args:
        current: LeetCodeState
            Current state before destroying
        random_state: random_state
            Random state for reproducibility
        destroy_factor: float
            Percentage of problems to remove
    Returns:
        destroyed: LeetCodeState
            State after destroying
    """
    destroyed = current.copy()
    
    if not destroyed.selected_problems:
        return destroyed
    
    # Get current difficulty distribution
    current_dist = destroyed._get_current_difficulty_distribution()
    target_dist = destroyed.difficulty_distribution[destroyed.skill_level]
    
    # Find difficulty that needs adjustment
    difficulties = ['Easy', 'Medium', 'Hard']
    difficulty_scores = [
        (diff, current_dist[diff] - target_dist[diff])
        for diff in difficulties
    ]
    
    # Bias towards removing easier problems
    if random_state.random() < 0.9:  # 70% chance to focus on easier problems
        # Find the easiest difficulty that has problems
        # for diff in ['Easy', 'Medium']:
        for diff in ['Easy']:
            if current_dist[diff] > 0:
                target_difficulty = diff
                break
        else:
            target_difficulty = max(difficulty_scores, key=lambda x: x[1])[0]
    else:
        # Select difficulty with highest deviation from target
        target_difficulty = max(difficulty_scores, key=lambda x: x[1])[0]
    
    # Find problems of target difficulty
    difficulty_problems = [
        problem for problem in destroyed.selected_problems
        if problem.difficulty == target_difficulty
    ]
    
    if not difficulty_problems:
        return destroyed
    
    # Determine number of problems to remove
    num_to_remove = max(1, int(len(difficulty_problems) * destroy_factor))
    num_to_remove = min(num_to_remove, len(difficulty_problems))
    
    # Randomly select problems to remove
    indices = random_state.choice(len(difficulty_problems), num_to_remove, replace=False)
    
    # Store problem IDs to remove
    problem_ids_to_remove = []
    for idx in indices:
        problem = difficulty_problems[idx]
        problem_ids_to_remove.append(problem.id)
    
    # Remove problems by ID (more reliable than by object)
    for problem_id in problem_ids_to_remove:
        destroyed.remove_problem(problem_id)
    
    return destroyed

def destroy_company_focused(current: LeetCodeState, random_state, destroy_factor=0.3):
    """Company-focused destroy operator
    
    Removes problems from specific companies to improve company coverage
    
    Args:
        current: LeetCodeState
            Current state before destroying
        random_state: random_state
            Random state for reproducibility
        destroy_factor: float
            Percentage of problems to remove
    Returns:
        destroyed: LeetCodeState
            State after destroying
    """
    destroyed = current.copy()
    
    if not destroyed.selected_problems:
        return destroyed
    
    # Get current company coverage
    current_companies = set()
    for problem in destroyed.selected_problems:
        current_companies.update(problem.companies)
    
    # Find companies that need more coverage
    target_companies = set(destroyed.target_companies)
    underrepresented_companies = target_companies - current_companies
    
    if not underrepresented_companies:
        return destroyed
    
    # Select a random underrepresented company
    target_company = random_state.choice(list(underrepresented_companies))
    
    # Find problems from other companies that could be replaced
    replaceable_problems = [
        problem for problem in destroyed.selected_problems
        if target_company not in problem.companies
    ]
    
    if not replaceable_problems:
        return destroyed
    
    # Determine number of problems to remove
    num_to_remove = max(1, int(len(replaceable_problems) * destroy_factor))
    num_to_remove = min(num_to_remove, len(replaceable_problems))
    
    # Randomly select problems to remove
    indices = random_state.choice(len(replaceable_problems), num_to_remove, replace=False)
    
    # Store problem IDs to remove
    problem_ids_to_remove = []
    for idx in indices:
        problem = replaceable_problems[idx]
        problem_ids_to_remove.append(problem.id)
    
    # Remove problems by ID (more reliable than by object)
    for problem_id in problem_ids_to_remove:
        destroyed.remove_problem(problem_id)
    
    return destroyed

### Repair operators ###

def topic_coverage_repair(destroyed: LeetCodeState, random_state):
    """Topic coverage repair operator
    
    Repairs solution by focusing on covering missing topics
    
    Args:
        destroyed: LeetCodeState
            State after destroying
        random_state: random_state
            Random state for reproducibility
    Returns:
        repaired: LeetCodeState
            State after repairing
    """
    repaired = destroyed.copy()
    
    # Find uncovered topics
    covered_topics = repaired.covered_topics
    all_topics = set()
    for problem in repaired.problems:
        all_topics.update(problem.topics)
    uncovered_topics = all_topics - covered_topics
    
    if not uncovered_topics:
        return repaired
    
    # For each uncovered topic, find problems that cover it
    topic_problems = {}
    for topic in uncovered_topics:
        topic_problems[topic] = [
            problem for problem in repaired.problems
            if topic in problem.topics and problem.id not in repaired.selected_problem_ids
        ]
    
    # Sort topics by number of available problems
    sorted_topics = sorted(
        uncovered_topics,
        key=lambda x: len(topic_problems[x]),
        reverse=True
    )
    
    # Try to cover each topic
    for topic in sorted_topics:
        available_problems = topic_problems[topic]
        if not available_problems:
            continue
        
        # Sort problems by number of additional topics they cover
        problem_scores = []
        for problem in available_problems:
            new_topics = set(problem.topics) - repaired.covered_topics
            score = len(new_topics)
            problem_scores.append((problem, score))
        
        # Try to add the best problem
        problem_scores.sort(key=lambda x: x[1], reverse=True)
        for problem, _ in problem_scores:
            if repaired.can_add_problem(problem):
                repaired.add_problem(problem)
                break
    
    return repaired

def greedy_repair(destroyed: LeetCodeState, random_state):
    """Greedy repair operator
    
    Repairs solution by greedily adding problems with highest objective contribution
    
    Args:
        destroyed: LeetCodeState
            State after destroying
        random_state: random_state
            Random state for reproducibility
    Returns:
        repaired: LeetCodeState
            State after repairing
    """
    repaired = destroyed.copy()
    
    # Calculate objective contribution for each available problem
    problem_scores = []
    for problem in repaired.problems:
        if problem.id in repaired.selected_problem_ids:
            continue
        
        # Create temporary state to evaluate problem
        temp_state = repaired.copy()
        if temp_state.can_add_problem(problem):
            temp_state.add_problem(problem)
            score = temp_state.objective() - repaired.objective()
            problem_scores.append((problem, score))
    
    # Sort problems by objective contribution
    problem_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Try to add problems in order of contribution
    for problem, _ in problem_scores:
        if repaired.can_add_problem(problem):
            repaired.add_problem(problem)
    
    return repaired

def company_focused_repair(destroyed: LeetCodeState, random_state):
    """Company-focused repair operator
    
    Repairs solution by focusing on target company coverage
    
    Args:
        destroyed: LeetCodeState
            State after destroying
        random_state: random_state
            Random state for reproducibility
    Returns:
        repaired: LeetCodeState
            State after repairing
    """
    repaired = destroyed.copy()
    
    # Get current company coverage
    current_companies = set()
    for problem in repaired.selected_problems:
        current_companies.update(problem.companies)
    
    # Find missing target companies
    missing_companies = set(repaired.target_companies) - current_companies
    
    if not missing_companies:
        return repaired
    
    # For each missing company, find problems from that company
    company_problems = {}
    for company in missing_companies:
        company_problems[company] = [
            problem for problem in repaired.problems
            if company in problem.companies and problem.id not in repaired.selected_problem_ids
        ]
    
    # Sort companies by number of available problems
    sorted_companies = sorted(
        missing_companies,
        key=lambda x: len(company_problems[x]),
        reverse=True
    )
    
    # Try to cover each company
    for company in sorted_companies:
        available_problems = company_problems[company]
        if not available_problems:
            continue
        
        # Sort problems by objective contribution
        problem_scores = []
        for problem in available_problems:
            temp_state = repaired.copy()
            if temp_state.can_add_problem(problem):
                temp_state.add_problem(problem)
                score = temp_state.objective() - repaired.objective()
                problem_scores.append((problem, score))
        
        # Try to add the best problem
        problem_scores.sort(key=lambda x: x[1], reverse=True)
        for problem, _ in problem_scores:
            if repaired.can_add_problem(problem):
                repaired.add_problem(problem)
                break
    
    return repaired

def acceptance_rate_repair(destroyed: LeetCodeState, random_state):
    """Acceptance rate repair operator
    
    Repairs solution by focusing on problems with high acceptance rates
    
    Args:
        destroyed: LeetCodeState
            State after destroying
        random_state: random_state
            Random state for reproducibility
    Returns:
        repaired: LeetCodeState
            State after repairing
    """
    repaired = destroyed.copy()
    
    # Find available problems with high acceptance rates
    available_problems = [
        problem for problem in repaired.problems
        if problem.id not in repaired.selected_problem_ids
    ]
    
    if not available_problems:
        return repaired
    
    # Sort problems by acceptance rate
    problem_scores = []
    for problem in available_problems:
        if repaired.can_add_problem(problem):
            temp_state = repaired.copy()
            temp_state.add_problem(problem)
            score = temp_state.objective() - repaired.objective()
            problem_scores.append((problem, score))
    
    # Sort by objective contribution
    problem_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Try to add problems in order of contribution
    for problem, _ in problem_scores:
        if repaired.can_add_problem(problem):
            repaired.add_problem(problem)
    
    return repaired

def popularity_repair(destroyed: LeetCodeState, random_state):
    """Popularity repair operator
    
    Repairs solution by focusing on popular problems
    
    Args:
        destroyed: LeetCodeState
            State after destroying
        random_state: random_state
            Random state for reproducibility
    Returns:
        repaired: LeetCodeState
            State after repairing
    """
    repaired = destroyed.copy()
    
    # Find available popular problems
    available_problems = [
        problem for problem in repaired.problems
        if problem.id not in repaired.selected_problem_ids
    ]
    
    if not available_problems:
        return repaired
    
    # Sort problems by popularity score
    problem_scores = []
    for problem in available_problems:
        if repaired.can_add_problem(problem):
            temp_state = repaired.copy()
            temp_state.add_problem(problem)
            score = temp_state.objective() - repaired.objective()
            problem_scores.append((problem, score))
    
    # Sort by objective contribution
    problem_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Try to add problems in order of contribution
    for problem, _ in problem_scores:
        if repaired.can_add_problem(problem):
            repaired.add_problem(problem)
    
    return repaired

def difficulty_repair(destroyed: LeetCodeState, random_state):
    """Difficulty repair operator
    
    Repairs solution by focusing on improving the difficulty score by prioritizing harder problems.
    This operator specifically targets the difficulty component of the objective function.
    
    Args:
        destroyed: LeetCodeState
            State after destroying
        random_state: random_state
            Random state for reproducibility
    Returns:
        repaired: LeetCodeState
            State after repairing
    """
    repaired = destroyed.copy()
    
    # Get current difficulty distribution
    current_dist = repaired._get_current_difficulty_distribution()
    target_dist = repaired.difficulty_distribution[repaired.skill_level]
    
    # Find available problems
    available_problems = [
        problem for problem in repaired.problems
        if problem.id not in repaired.selected_problem_ids
    ]
    
    if not available_problems:
        return repaired
    
    # Calculate difficulty scores for each problem
    problem_scores = []
    for problem in available_problems:
        if repaired.can_add_problem(problem):
            # Create temporary state to evaluate problem
            temp_state = repaired.copy()
            temp_state.add_problem(problem)
            
            # Calculate difficulty score based on problem difficulty
            if problem.difficulty == 'Hard':
                difficulty_score = 1.0
            elif problem.difficulty == 'Medium':
                difficulty_score = 0.6
            else:  # Easy
                difficulty_score = 0.2
            
            # Calculate objective contribution
            objective_diff = temp_state.objective() - repaired.objective()
            
            # Combine difficulty score with objective contribution
            # Give more weight to difficulty score for harder problems
            combined_score = (difficulty_score * 0.7) + (objective_diff * 0.3)
            problem_scores.append((problem, combined_score))
    
    # Sort problems by combined score
    problem_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Try to add problems in order of score
    for problem, _ in problem_scores:
        if repaired.can_add_problem(problem):
            repaired.add_problem(problem)
    
    return repaired
