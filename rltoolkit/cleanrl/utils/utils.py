from typing import Dict, List


def calculate_mean(dict_list: List[Dict[str, float]]) -> Dict[str, float]:
    """计算包含字典的列表中每个键的均值。

    Args:
        dict_list (List[Dict[str, float]]): 包含字典的列表。

    Returns:
        Dict[str, float]: 每个键的均值字典。
    """
    # 检查是否为空列表
    if not dict_list:
        return {}

    # 初始化累加器字典
    sum_dict = {key: 0 for key in dict_list[0].keys()}

    # 累加每个字典中的值
    for d in dict_list:
        for key, value in d.items():
            sum_dict[key] += value

    # 计算均值
    mean_dict = {
        key: total / len(dict_list)
        for key, total in sum_dict.items()
    }

    return mean_dict
