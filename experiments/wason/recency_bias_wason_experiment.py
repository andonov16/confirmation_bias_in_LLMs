import random
import os
import pandas as pd

from src.wason_selection import conduct_wason_selection_task
from src.model_clients.base_model_client import BaseModelClient


def conduct_recency_bias_wason_selection_task_experiment(
    tested_llm_model: str,
    model_client: BaseModelClient,
    wason_tasks_df: pd.DataFrame,
    results_log_dir: str,
    walson_config: dict,
) -> pd.DataFrame:
    random.seed(walson_config['experiment_seed'])
    llm_task = walson_config['control_llm_task']
    results = {
        'Model': [],
        'Rule': [],
        'P': [],
        'not P': [],
        'Q': [],
        'not Q': [],
        'Picked Items': [],
        'Solved Correctly': []
    }

    for index, row in wason_tasks_df.iterrows():
        items = {
            'P': row['P'],
            'not P': row['not P'],
            'Q': row['Q'],
            'not Q': row['not Q'],
        }

        items_ordering = ['not P', 'Q', 'not Q', 'P']

        solved_correctly, picked_items = conduct_wason_selection_task(
            model_client=model_client,
            rule=row['Rule'],
            items=items,
            items_ordering=items_ordering,
            llm_task=llm_task
        )

        results['Model'].append(tested_llm_model)
        results['Rule'].append(row['Rule'])
        results['P'].append(row['P'])
        results['not P'].append(row['not P'])
        results['Q'].append(row['Q'])
        results['not Q'].append(row['not Q'])
        results['Picked Items'].append(picked_items)
        results['Solved Correctly'].append(solved_correctly)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_log_dir, f'{tested_llm_model}_recency_bias_wason_study_results.csv'), index=False)

    return results_df