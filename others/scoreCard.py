# -*- coding: utf-8 -*-
from math import log


def scoreCard(rawprediction, P=800, PDO=60, ratio=0.02):
    """
    probability: 违约概率
    ratio: 违约比率
    rawprediction: 回归公式计算结果
    score: 模型评分
    P = A - B * ln(ratio)
    P - PDO = A - B * ln(2 * ratio)
    probability = 1/(1 + e^(-rawprediction)) =>
    rawprediction = ln(probability/(1-probability)) = ln(ratio)

    461.36862861351653, 86.5617024533378
    """
    A = P + PDO * log(ratio) / log(2)
    B = PDO / log(2)
    score = A - B * rawprediction
    print(A, B)
    return score


# var = [-1.0181143663164438, -0.9800066988655683, -0.8572897613925428, -0.7332909183665288, -0.7599095815994613,
#        -0.8911541190722109, -0.8336264602559279]
# coefficients = [-0.36467, -0.47606, -0.58566, -0.33945, -0.20803, 0.16336, -0.49099, 0.01175]
# score = coefficients[-1]
# for r, coef in zip(var, coefficients[:-1]):
#     score = score + r * coef
# print(score)

print(scoreCard(-1.1650872021566787))
print(scoreCard(2.194763075360513))
