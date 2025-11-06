import random
import pandas as pd
from pandas.core.computation.common import result_type_many

from src.wason_selection import conduct_wason_selection_task
from src.model_clients.base_model_client import BaseModelClient


def conduct_anchoring_wason_selection_task_experiment(
    tested_llm_model: str,
    model_client: BaseModelClient,
    wason_tasks_df: pd.DataFrame,
    walson_config: dict,
) -> pd.DataFrame:
    random.seed(walson_config['experiment_seed'])
    llm_task = walson_config['control_llm_task']

    results = {
        'Intentional Bias': [],
        'Model': [],
        'Rule': [],
        'P': [],
        'not P': [],
        'Q': [],
        'not Q': [],
        'Picked Items': [],
        'Solved Correctly': [],
        'User Prompt': [],
        'Raw LLM Output': []
    }

    for index, row in wason_tasks_df.iterrows():
        items = {
            'P': row['P'],
            'not P': row['not P'],
            'Q': row['Q'],
            'not Q': row['not Q'],
        }

        items_ordering = list(items.keys())
        random.shuffle(items_ordering)

        p, q = items['P'], items['Q']
        
        solved_correctly, picked_items, raw_llm_output, raw_user_prompt = conduct_wason_selection_task(
            model_client=model_client,
            rule=row['Rule'],
            items=items,
            items_ordering=items_ordering,
            llm_task=f'{llm_task} "{p}" and "{q}"'
        )

        results['Intentional Bias'].append('Anchoring')
        results['Model'].append(tested_llm_model)
        results['Rule'].append(row['Rule'])
        results['P'].append(row['P'])
        results['not P'].append(row['not P'])
        results['Q'].append(row['Q'])
        results['not Q'].append(row['not Q'])
        results['Picked Items'].append(picked_items)
        results['Solved Correctly'].append(solved_correctly)
        results['User Prompt'].append(raw_user_prompt)
        results['Raw LLM Output'].append(raw_llm_output)

    results_df = pd.DataFrame(results)
    return results_df