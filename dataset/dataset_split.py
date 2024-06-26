import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


def generate_splits(target_path, seed, subset='dropGradNaN'):
    data_cv = pickle.load(open("./data/splits/gbmlgg15cv_all_st_0_0_0.pkl", 'rb'))
    all_data = pd.read_csv("./data/TCGA_GBMLGG/all_dataset.csv")
    all_grade = pd.read_csv("./data/TCGA_GBMLGG/grade_data.csv")

    no_mole_pat = all_grade[all_grade['Molecular subtype'].isna()]['TCGA ID'].values
    no_grade_pat = all_grade[all_grade['Grade'].isna()]['TCGA ID'].values
    no_mole_grade = np.concatenate([no_mole_pat, no_grade_pat])
    if subset == 'dropGradNaN':
        target_list = no_grade_pat
    elif subset == 'dropGradGeneNaN':
        target_list = no_mole_grade
    else:
        raise ValueError("fileset should be 'no_grade' or 'no_grade_no_gene'")

    train_path = data_cv['cv_splits'][1]['train']['x_path']
    test_path = data_cv['cv_splits'][1]['test']['x_path']
    train_grade = data_cv['cv_splits'][1]['train']['g']
    test_grade = data_cv['cv_splits'][1]['test']['g']
    train_omic = data_cv['cv_splits'][1]['train']['x_omic']
    test_omic = data_cv['cv_splits'][1]['test']['x_omic']
    train_patname = data_cv['cv_splits'][1]['train']['x_patname']
    test_patname = data_cv['cv_splits'][1]['test']['x_patname']

    path = np.concatenate([train_path, test_path])
    omic = np.concatenate([train_omic, test_omic])
    grade = np.concatenate([train_grade, test_grade])
    patname = np.concatenate([train_patname, test_patname])

    idx = [i for i in range(len(grade)) if patname[i] not in target_list]
    my_path = path[idx]
    my_omic = omic[idx]
    my_grade = grade[idx]

    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    my_data = dict()
    my_data['splits'] = dict()
    my_data['data_pd'] = data_cv['data_pd']
    for i, (train_id, test_id) in enumerate(kf.split(my_grade)):
        temp = dict()
        temp['train'] = dict()
        temp['test'] = dict()
        temp['train']['x_path'] = my_path[train_id]
        temp['train']['x_omic'] = my_omic[train_id]
        temp['train']['grade'] = my_grade[train_id]

        temp['test']['x_path'] = my_path[test_id]
        temp['test']['x_omic'] = my_omic[test_id]
        temp['test']['grade'] = my_grade[test_id]
        my_data['splits'][i] = temp

    with open(target_path, "wb") as f:
        pickle.dump(my_data, f)

    print(f"Split the dataset with {subset} and save to {target_path}")


if __name__ == '__main__':
    generate_splits("./dataset/my_split_dropGradeNaN.pkl", 42)
