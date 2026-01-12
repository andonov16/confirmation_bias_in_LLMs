import pandas as pd
from tqdm import tqdm

from src.wason_selection import conduct_wason_selection_task
from src.model_clients.base_model_client import BaseModelClient


def conduct_wason_selection_task_experiment(
        tested_llm_model: str,
        model_client: BaseModelClient,
        wason_tasks_df: pd.DataFrame,
        wason_experiment_config: dict,
) -> pd.DataFrame:
    llm_task = wason_experiment_config['llm_task']

    # printing explanatory console messages
    print('='*100)
    print(wason_experiment_config['console_log_message'])
    print('LLM Model: ', tested_llm_model)


    results = {
        'Task ID': [],
        'Experiment Type': [],
        'LLM Task': [],
        'Rule': [],
        'P': [],
        'not P': [],
        'Q': [],
        'not Q': [],
        'Cards Ordering': [],
    }

    for token in wason_experiment_config['allowed_tokens']:
        results[f'{token} DCPMI'] = []


    for index, row in tqdm(wason_tasks_df.iterrows(), desc='Conducted Wason Tasks'):
        results['Task ID'].append(row['Task ID'])
        results['Experiment Type'].append(wason_experiment_config['experiment_type'])
        results['LLM Task'].append(llm_task)
        results['Rule'].append(row['Rule'])
        results['P'].append(row['P'])
        results['not P'].append(row['not P'])
        results['Q'].append(row['Q'])
        results['not Q'].append(row['not Q'])
        results['Cards Ordering'].append(wason_experiment_config['cards_ordering'])


        items = {
            'P': row['P'],
            'not P': row['not P'],
            'Q': row['Q'],
            'not Q': row['not Q'],
        }

        # conducting the wason experiment using the LLM`s API
        try:
            llm_probs_result_dict = conduct_wason_selection_task(
                model_client=model_client,
                rule=row['Rule'],
                items=items,
                items_ordering=wason_experiment_config['cards_ordering'],
                llm_task=llm_task,
                allowed_tokens=wason_experiment_config['allowed_tokens'],
            )

            for allowed_token, dcpmi_val in llm_probs_result_dict.items():
                results[f'{allowed_token} DCPMI'].append(dcpmi_val)
        except RuntimeError:
            for allowed_token in wason_experiment_config['allowed_tokens']:
                results[f'{allowed_token} DCPMI'].append('NaN')

        print(results)


    results_df = pd.DataFrame(results)
    return results_df