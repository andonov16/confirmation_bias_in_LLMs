import os
import yaml
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Dict

from src.model_clients.open_ai_client import OpenAIClient
from src.model_clients.ollama_ai_client import OllamaAIClient

# Import your experiment
from experiments.base_wason_experiment import conduct_wason_selection_task_experiment

def conduct_wason_study(api_config_path: str,
                        wason_config_paths: Dict[str, str],
                        results_dir: str,
                        wason_tasks_df: pd.DataFrame):
    """
    Runs all four Wason Selection Task studies (Control, Framing, Anchoring, Recency)
    across all LLM configurations listed in the given api_config YAML.

    Parameters
    ----------
    api_config_path : str
        Path to the YAML file listing API keys, base URLs, and model names.



    results_dir : str
        Directory to store results CSVs. Created if not existing.
    wason_tasks_df : pd.DataFrame
        A dataframe containing the Wason tasks (columns: Rule, P, not P, Q, not Q).
    """
    # Load configs
    with open(api_config_path, 'r') as f:
        api_config = yaml.safe_load(f)

    wason_config_dicts: Dict[str, dict] = {}
    for experiment_name, wason_config_path in wason_config_paths.items():
        with open(wason_config_path, 'r') as f:
            wason_config_dicts[experiment_name] = yaml.safe_load(f)

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # --- Iterate through each model in the API config ---
    for model_info in tqdm(api_config['LLMs_to_test'], desc='Remaining models'):
        api_key = model_info['api_key']
        base_url = model_info['base_url']
        model_name = model_info['model_name']
        model_type = model_info["model_type"]

        model_client = None

        if model_type == 'local':
            # Create model client
            model_client = OllamaAIClient(
                api_key=api_key,
                base_url=base_url,
                model_name=model_name
            )
        else:
            # Create model client
            model_client = OpenAIClient(
                api_key=api_key,
                base_url=base_url,
                model_name=model_name
            )

        # Run all 4 studies
        for experiment_name, experiment_config_dict in wason_config_dicts.items():
            result_path = os.path.join(results_dir,
                                       f'{model_name.replace(":", "_")}_{experiment_name}_wason_study_results.csv')  # Can't have ':' in filenames
            # print(result_path)

            if os.path.exists(result_path):
                print(f'Skipping {experiment_name} study for {model_name} (already exists).')
                continue

            # print(f'\nRunning {bias_name.capitalize()} study for {model_name}...')

#            try:
            results_df = conduct_wason_selection_task_experiment(
                tested_llm_model=model_name,
                model_client=model_client,
                wason_tasks_df=wason_tasks_df,
                wason_experiment_config=experiment_config_dict
            )

            # Ensure consistent save
            results_df.to_csv(result_path, index=False)
            print(f'Saved results to {result_path}')

#            except Exception as e:
#                print(f' Failed {experiment_name} study for {model_name}: {e}')

    print('\nAll Wason studies completed!')


# Example usage:
if __name__ == '__main__':
    import pandas as pd

    root_path = Path(__file__).resolve().parent.parent  # Project root

    test_wason_config_paths: Dict[str, str] = {
        'anchoring': str(root_path / 'config/wason_anchoring_config.yaml'),
        'control': str(root_path / 'config/wason_control_config.yaml'),
        'framing': str(root_path / 'config/wason_framing_config.yaml'),
        'position': str(root_path / 'config/wason_position_config.yaml')
    }

    conduct_wason_study(
        api_config_path=str(root_path / 'config/api_config.yaml'),
        wason_config_paths=test_wason_config_paths,
        results_dir=str(root_path / 'logs'),
        wason_tasks_df=pd.read_csv(str(root_path / 'data/wason_selection_tasks.csv'))
    )
