import time
from z3 import *

def solve_student_group(students_df, group_size):
    solver = Optimize()
    num_candidates = len(students_df)
    x = [Int(f"x_{i}") for i in range(num_candidates)]
    
    # 1. Binary & Size
    for i in range(num_candidates): solver.add(Or(x[i] == 0, x[i] == 1))
    solver.add(Sum(x) == group_size)
    
    # 2. Gender (Hard Constraint)
    genders = students_df['Gender'].values
    min_males = int(group_size * 0.4)
    max_males = int(group_size * 0.6)
    male_terms = [x[i] for i in range(num_candidates) if genders[i] > 0.5]
    solver.add(Sum(male_terms) >= min_males)
    solver.add(Sum(male_terms) <= max_males)
    
    # 3. OBJECTIVE (Maximize ALL Features)    
    exams = students_df['ExamScore'].values
    assigns = students_df['AssignmentCompletion'].values
    attends = students_df['Attendance'].values
    discs = students_df['Discussions'].values
    edutech = students_df['EduTech'].values
    extras = students_df['Extracurricular'].values
    
    # Weights match RL logic approx.
    # Exam(0.7) + Assign(0.3) + Attend(0.2) + Disc(0.1) + Edu(0.1) + Extra(0.1)
    weighted_scores = []
    for i in range(num_candidates):
        score = (exams[i]*700) + (assigns[i]*300) + \
                (attends[i]*200) + (discs[i]*100) + \
                (edutech[i]*100) + (extras[i]*100)
        weighted_scores.append(int(score))
    
    objective = Sum([x[i] * weighted_scores[i] for i in range(num_candidates)])
    solver.maximize(objective)
    
    if solver.check() == sat:
        model = solver.model()
        return [i for i in range(num_candidates) if model[x[i]] == 1]
    else:
        return []

def run_smt(df_test, group_size=10):
    print("\nClustering with SMT (Z3)...")
    start_time = time.time()
    ungrouped = df_test.copy().reset_index(drop=True)
    groups = []
    while len(ungrouped) >= group_size:
        indices = solve_student_group(ungrouped, group_size)
        if len(indices) == group_size:
            groups.append(ungrouped.iloc[indices].values)
            ungrouped = ungrouped.drop(indices).reset_index(drop=True)
        else:
            break
    if len(ungrouped) > 0: groups.append(ungrouped.values)
    return groups, time.time() - start_time
