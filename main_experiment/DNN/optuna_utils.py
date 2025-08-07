




def study_list_from_all_study_summaries(all_study_summaries, min_completed_trials = 50):
    """
    Given a list of study summaries, return a list of studies
    """
    study_list = []
    for study_summary in all_study_summaries:
        if study_summary.n_trials > min_completed_trials:
            study = study_summary.study
            study_list.append(study)
    #if modify first block is not provided, just put in "None" (my bad)
    # Actually, modifying the first block might be okay...? Or not? (Can this be ignored?)
    return study_list