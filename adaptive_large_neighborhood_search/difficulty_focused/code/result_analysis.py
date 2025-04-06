import pandas as pd
from collections import Counter
import ast

# Load the study plan data
df = pd.read_csv(r'mathematical_programming\result\study_plan.csv')

# Basic overview
total_days = df['day'].max()
total_problems = len(df)
print(f"**Overview**")
print(f"* Study Plan: {total_days} days, {total_problems} problems")

# Time utilization
total_time = df['estimated_time'].sum()
max_time = total_days * 120  # assuming 120 minutes per day is the maximum
time_utilization = (total_time / max_time) * 100
print(f"* Time Utilization: {time_utilization:.1f}% of maximum study time")

# Count unique topics
all_topics = set()
for topics_str in df['topics']:
    topics_list = ast.literal_eval(topics_str)
    all_topics.update(topics_list)
print(f"* Topic Coverage: {len(all_topics)} core topics fully covered")

# Difficulty distribution
difficulty_counts = df['difficulty'].value_counts(normalize=True) * 100
print("* Difficulty Distribution:")
for difficulty, percentage in difficulty_counts.items():
    print(f"   * {difficulty}: {percentage:.0f}%")

# Process all companies
all_companies = []
for companies_str in df['companies']:
    companies_list = ast.literal_eval(companies_str)
    all_companies.extend(companies_list)

# Count companies
company_counter = Counter(all_companies)
print("**Company Coverage**")
for company, count in [('Amazon', company_counter['Amazon']), 
                       ('Google', company_counter['Google']), 
                       ('Microsoft', company_counter['Microsoft'])]:
    percentage = (count / total_problems) * 100
    print(f"* {company}: {count} problems ({percentage:.0f}%)")

# Process all topics
all_topics_list = []
for topics_str in df['topics']:
    topics_list = ast.literal_eval(topics_str)
    all_topics_list.extend(topics_list)

# Count topics and get top 3
topic_counter = Counter(all_topics_list)
print("**Top Topics**")
for topic, count in topic_counter.most_common(3):
    print(f"* {topic} ({count} problems)")