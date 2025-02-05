def jaccard(a: float, b: float, intersection: float) -> float:
    """Calculate the Jaccard index:
    J(A,B) = |A∩B|/|A∪B|"""
    return intersection / (a + b - intersection)


def tversky(
    a_minus_b: float, b_minus_a: float, intersection: float, alpha: float, beta: float
) -> float:
    """Calculate the Tversky index:
    S(A,B) = |A∩B|/(|A∩B| + α|A-B| + β|B-A|)"""
    return intersection / (intersection + alpha * a_minus_b + beta * b_minus_a)
