from ast import literal_eval
from typing import Union


def format_span_list(x: Union[str, list, None]) -> list:
    """This function takes a string or a list of strings and returns a list of strings.
    
    Parameters:
        x (str or list): A string or a list of strings.
    
    Returns:
        list: A list of strings.
    """
    
    try:
        x = literal_eval(x)
    except:
        pass
    if isinstance(x, list): return x
    elif isinstance(x, str): return [x]
    else: return []
    

def generate_span_labels(ELLIPTICAL_TYPES, SPAN_TYPE):
    
    if SPAN_TYPE == 'discriminative':
        label_list = [ "O", "B", "I" ]

    if SPAN_TYPE == 'extractive':
        label_list = [ "O" ]
        _ = [ label_list.extend([ f'B-{label}', f'I-{label}' ]) for label in ELLIPTICAL_TYPES]
        
    id2label = dict(enumerate(label_list))
    label2id = { label: _id for _id, label in id2label.items()}