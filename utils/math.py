from typing import List


def math_expectation(indexed_messages: List[List[int]]) -> float:
    math_exp = 0
    for messages in indexed_messages:
        math_exp += len(messages)
    return math_exp / len(indexed_messages)


def dispersion(indexed_messages: List[List[int]], math_exp: float) -> float:
    dispersion = 0
    for messages in indexed_messages:
        dispersion += (len(messages) - math_exp) ** 2
    return dispersion / len(indexed_messages)
