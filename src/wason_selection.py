from src.model_clients.base_model_client import BaseModelClient
from typing import Dict,List


def conduct_wason_selection_task(
        model_client: BaseModelClient,
        rule: str,
        items: Dict[str, str],
        items_ordering: List[str],
        llm_task: str,
        allowed_tokens: List[str],
) -> Dict[str, float]:
    """
    Parameters
    ----------
        :param model_client : BaseModelClient
            Wrapper object that handles the prompt sending and getting a response from the specified model
        :param rule: str
            The rule to be checked by the model to test its reasoning. Must be of the form "If P, then Q."
            e.g. "If a card has a vowel on one side, then it has an even number on the other side."
        :param items: Dict[str, str]:
            Dictionary that contains all the possible cards from which one has to pick to determine if the rule is true or false.
            Form:
                Key: "P", Value: "The card has a vowel on one side."
                Key: "not P", Value: "Card has a consonant on one side."
                Key: "Q": "The card has an even number."
                Key: "not Q": "The card has an odd number."
       :param  items_ordering: List[str]:
            List of cards ordered by what order they will be presented to the model. e.g.['P', 'not P', 'Q', 'not Q']
        :param llm_task: str
            Defines the task given to the LLM model (the system prompt). Here anchoring can be introduced!
            e.g. "You are given a conditional rule and a set of items. Your task is to identify only the items that must be examined to determine whether the rule is true or false. Output only the text of the selected items, separated by semicolons (;).
            Do not include any reasoning, explanations, numbers, or extra text."
        :param allowed_tokens:

    Returns (Dict[str,float]):
    Dictionary that contains all the possible flipped cards combinations and their respective DCPMI values.


    """
    user_prompt = f'Rule: {rule}\nItems:\n'
    for i, item_key in zip(['A', 'B', 'C', 'D'], items_ordering):
        user_prompt += f'{i}. "{items[item_key]}"\n'

    # Get model response
    baseline_response_probs = model_client.send_prompt_get_response(
        system_prompt=llm_task,
        user_prompt='A.\n B.\nC.\nD.',
        allowed_tokens=allowed_tokens
    )

    rule_introduced_response_probs = model_client.send_prompt_get_response(
        system_prompt=llm_task,
        user_prompt=user_prompt,
        allowed_tokens=allowed_tokens
    )

    dcpmi_tokens_dict = {}
    for token in allowed_tokens:
        if token in rule_introduced_response_probs.keys() and token in baseline_response_probs.keys():
            dcpmi_tokens_dict[token] = rule_introduced_response_probs[token]/baseline_response_probs[token]
        elif token in rule_introduced_response_probs.keys():
            dcpmi_tokens_dict[token] = rule_introduced_response_probs[token]/1e-6
        else:
            dcpmi_tokens_dict[token] = float('-inf')

    return dcpmi_tokens_dict