import math
import torch
from typing import Any, Dict, List, Optional, Union


def _normalize_pos_entry(entry: Union[Dict[str, Any], List[float], tuple]) -> List[float]:
    """
    Convert a pose entry into a flat [x, y, theta] list.
    """
    if isinstance(entry, dict):
        if all(k in entry for k in ("x", "y", "theta")):
            return [entry["x"], entry["y"], entry["theta"]]
        if "pos" in entry:
            return _normalize_pos_entry(entry["pos"])
        if "pos_values" in entry:
            return _normalize_pos_entry(entry["pos_values"])
    elif isinstance(entry, (list, tuple)):
        if len(entry) == 3 and all(isinstance(v, (int, float)) for v in entry):
            return list(entry)
    raise ValueError(f"Unsupported pose entry format: {entry}")


def extract_pos_info(conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]) -> List[List[float]]:
    """
    Extract pose entries from conversations. Looks for elements with `type == "pos"` or keys
    like `pos`, `pos_values`, or dicts containing `x`, `y`, `theta`.
    """
    pos_list: List[List[float]] = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message.get("content"), list):
                for ele in message["content"]:
                    if (
                        ele.get("type") == "pos"
                        or "pos" in ele
                        or "pos_values" in ele
                    ):
                        pos_list.append(_normalize_pos_entry(ele))
    return pos_list


def process_pos_info(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
) -> Optional[torch.Tensor]:
    """
    Process pose tokens from conversations into a tensor shaped (num_pose_tokens, 3).
    """
    pos_entries = extract_pos_info(conversations)
    if len(pos_entries) == 0:
        return None
    return torch.tensor(pos_entries, dtype=torch.float32)


def _normalize_act_sequence(act_seq: Any) -> List[List[float]]:
    """
    Normalize an action sequence into [[x, y, sin(theta), cos(theta)], ...].
    Accepts a list/tuple of triples or dicts with x/y/theta.
    """
    if act_seq is None:
        return []
    if not isinstance(act_seq, (list, tuple)):
        raise ValueError(f"Unsupported act sequence format: {act_seq}")

    normalized: List[List[float]] = []
    for item in act_seq:
        if isinstance(item, dict) and all(k in item for k in ("x", "y", "theta")):
            x, y, theta = item["x"], item["y"], item["theta"]
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            x, y, theta = item
        else:
            raise ValueError(f"Unsupported act entry format: {item}")
        theta = float(theta)
        normalized.append(
            # [float(x), float(y), math.sin(theta), math.cos(theta)]
            [float(x), float(y), float(theta)]
        )
    return normalized


def extract_act_info(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]
) -> List[List[List[float]]]:
    """
    Extract action sequences from conversations. Looks for elements with `type == "act"`.
    Returns a list of action sequences, one per conversation.
    """
    act_list: List[List[float]] = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message.get("content"), list):
                for ele in message["content"]:
                    if isinstance(ele, dict) and ele.get("type") == "act_nav":
                        seq = _normalize_act_sequence(ele.get("act_nav"))
                        act_list.append(seq)
    return act_list
    
    
    # if not conversations:
    #     return []
    # if isinstance(conversations[0], dict):
    #     conversations = [conversations]

    # act_sequences: List[List[List[float]]] = []
    # for conv in conversations:
    #     seq = []
    #     found = False
    #     for message in conv:
    #         if not isinstance(message.get("content"), list):
    #             continue
    #         for ele in message["content"]:
    #             if isinstance(ele, dict) and ele.get("type") == "act_nav":
    #                 seq = _normalize_act_sequence(ele.get("act_nav"))
    #                 found = True
    #                 break
    #         if found:
    #             break
    #     act_sequences.append(seq)
    # return act_sequences


def process_act_info(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]
) -> Optional[torch.Tensor]:
    """
    Process action sequences into a tensor of shape (batch, seqlen, 4),
    where each entry is [x, y, sin(theta), cos(theta)].
    Returns None if no valid action sequences are found.
    """
    act_sequences = extract_act_info(conversations)
    if len(act_sequences) == 0:
        return None

    return torch.tensor(act_sequences, dtype=torch.float32)
    
    # max_len = max((len(seq) for seq in act_sequences), default=0)
    # if max_len == 0:
    #     return None

    # batch_data: List[List[List[float]]] = []
    # for seq in act_sequences:
    #     if len(seq) == 0:
    #         padded = [[0.0, 0.0, 0.0, 1.0] for _ in range(max_len)]
    #     elif len(seq) < max_len:
    #         padded = seq + [[0.0, 0.0, 0.0, 1.0] for _ in range(max_len - len(seq))]
    #     else:
    #         padded = seq
    #     batch_data.append(padded)

    # return torch.tensor(batch_data, dtype=torch.float32)
