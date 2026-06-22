import pandas as pd

def load_and_clean_data(filepath):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print("CSV not found.")
        return None
        
    features = ['Attendance', 'Extracurricular', 'Gender', 'Discussions', 
                'AssignmentCompletion', 'ExamScore', 'EduTech']
    return df[features].copy()

def prepare_data(filepath, student_count, random_state):
    full_data = load_and_clean_data(filepath)
    if full_data is None:
        return None, None, None, None
        
    test_data_original = full_data.sample(n=student_count, random_state=random_state)
    train_data = full_data.drop(test_data_original.index)
    
    df_rl = test_data_original.copy()
    df_smt = test_data_original.copy()
    
    # Normalization (fitted on train_data only to prevent data leakage)
    for col in ['Attendance', 'AssignmentCompletion', 'ExamScore']:
        train_min = train_data[col].min()
        train_max = train_data[col].max()
        
        if train_max != train_min:
            train_data[col] = (train_data[col] - train_min) / (train_max - train_min)
            df_rl[col] = (df_rl[col] - train_min) / (train_max - train_min)
            df_smt[col] = (df_smt[col] - train_min) / (train_max - train_min)
            full_data[col] = (full_data[col] - train_min) / (train_max - train_min)
        else:
            train_data[col] = 0.0
            df_rl[col] = 0.0
            df_smt[col] = 0.0
            full_data[col] = 0.0
            
    print(f"Data Loaded - Student Set for Evaluation: {len(test_data_original)} Students")
    
    return full_data, train_data, df_rl, df_smt
