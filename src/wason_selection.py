from src.model_clients.base_model_client import BaseModelClient
from typing import Dict,List, Tuple


def conduct_wason_selection_task(
        model_client: BaseModelClient,
        rule: str,
        items: Dict[str, str],
        items_ordering: List[str],
        llm_task: str
) -> Tuple[bool, List[str]]:
    """
    Parameters
    ----------
        model_client : BaseModelClient
            Wrapper object that handles the prompt sending and getting a response from the specified model
        rule: str
            The rule to be checked by the model to test its reasoning. Must be of the form "If P, then Q."
            e.g. "If a card has a vowel on one side, then it has an even number on the other side."
        items: Dict[str, str]:
            Dictionary that contains all the possible cards from which one has to pick to determine if the rule is true or false.
            Form:
                Key: "P", Value: "The card has a vowel on one side."
                Key: "not P", Value: "Card has a consonant on one side."
                Key: "Q": "The card has an even number."
                Key: "not Q": "The card has an odd number."
        items_ordering: List[str]:
            List of cards ordered by what order they will be presented to the model. e.g.['P', 'not P', 'Q', 'not Q']
        llm_task: str
            Defines the task given to the LLM model (the system prompt). Here anchoring can be introduced!
            e.g. "You are given a conditional rule and a set of items. Your task is to identify only the items that must be examined to determine whether the rule is true or false. Output only the text of the selected items, separated by semicolons (;).
            Do not include any reasoning, explanations, numbers, or extra text."
    Returns (Tuple[bool, Dict[str, str]])
    ----------
            bool: Returns true if the task is done successfully i.e. the model picked P & not Q.
            List[str]: returns all the items picked by the model in the form from above (e.g. [P, not Q]

    """
    user_prompt = f'\nRule: {rule}\nItems:\n'
    for i, item_key in enumerate(items_ordering, start=1):
        user_prompt += f'{i}. "{items[item_key]}"\n'

    print(llm_task)
    print(user_prompt)
    # Get model response
    response = model_client.send_prompt_get_response(system_prompt=llm_task,
                                                     user_prompt=user_prompt)

    #response = 'The card has a vowel on one side.;The card has an odd number.'

    # Process response: split by semicolon and remove whitespace
    picked_items = [x.strip() for x in response.split(';') if x.strip() in items.values()]
    success = items['P'] in picked_items and items['not Q'] in picked_items

    return success, [key for key in items.keys() if items[key] in picked_items]


if __name__ == '__main__':
    import yaml
    from src.model_clients.open_ai_client import OpenAIClient

    with open('../config/api_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        # Example setup
        api_key = config['openai_api_key']  # replace with your key

    client = OpenAIClient(api_key=api_key, base_url='')

    # Define rule and items
    rule = 'If a card has a vowel on one side, then it has an even number on the other side.'
    items = {
        'P': 'The card has a vowel on one side.',
        'not P': 'The card has a consonant on one side.',
        'Q': 'The card has an even number.',
        'not Q': 'The card has an odd number.'
    }
    items_ordering = ['P', 'not P', 'Q', 'not Q']

    # Define LLM task/system prompt
    llm_task = (
        'You are given a conditional rule and a set of items.'
        'Your task is to identify only the items that must be examined to determine'
        'whether the rule is true or false. Output only the text of the selected items, '
        'separated by semicolons (;). Do not include any reasoning, explanations, numbers,'
        'or extra text.'
    )
    client = None
    # Run the Wason selection task
    success, picked_items = conduct_wason_selection_task(
        model_client=client,
        rule=rule,
        items=items,
        items_ordering=items_ordering,
        llm_task=llm_task
    )

    print('Success:', success)
    print('Picked items:', picked_items)
