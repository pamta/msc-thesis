# Script from http://veekaybee.github.io/how-big-of-a-sample-size-do-you-need/ on how to calculate sample size, adjusted for my own population size
# and confidence intervals
# Original here: http://bc-forensics.com/?p=15

import math
import pandas as pd

# SUPPORTED CONFIDENCE LEVELS: 50%, 68%, 90%, 95%, and 99%
confidence_level_constant = (
    [50, 0.67],
    [68, 0.99],
    [80, 1.28],
    [85, 1.44],
    [90, 1.64],
    [95, 1.96],
    [99, 2.57],
)

# CALCULATE THE SAMPLE SIZE
def sample_size(population_size, confidence_level, confidence_interval):
    Z = 0.0
    p = 0.5
    e = confidence_interval / 100.0
    N = population_size
    n_0 = 0.0
    n = 0.0

    # LOOP THROUGH SUPPORTED CONFIDENCE LEVELS AND FIND THE NUM STD
    # DEVIATIONS FOR THAT CONFIDENCE LEVEL
    for i in confidence_level_constant:
        if i[0] == confidence_level:
            Z = i[1]

    if Z == 0.0:
        return -1

    # CALC SAMPLE SIZE
    n_0 = ((Z ** 2) * p * (1 - p)) / (e ** 2)

    # ADJUST SAMPLE SIZE FOR FINITE POPULATION
    n = n_0 / (1 + ((n_0 - 1) / float(N)))

    return int(math.ceil(n))  # THE SAMPLE SIZE


sample_sz = 0
population_sz = 10031
confidence_level = 95
confidence_interval = 5.0

#sample_sz = sample_size(population_sz, confidence_level, confidence_interval)

#print(sample_sz)

# df = pd.read_csv("to_validate.csv")
# df.sample(n=383, random_state=42).index

df = pd.read_csv("res.csv")

df["sample_50_05"] = [sample_size(size, 50.0, 5.0) for size in df["count"]]
df["sample_50_10"] = [sample_size(size, 50.0, 10.0) for size in df["count"]]
df["sample_68_05"] = [sample_size(size, 68.0, 5.0) for size in df["count"]]
df["sample_68_10"] = [sample_size(size, 68.0, 10.0) for size in df["count"]]
df["sample_80_05"] = [sample_size(size, 80.0, 5.0) for size in df["count"]]
df["sample_80_10"] = [sample_size(size, 80.0, 10.0) for size in df["count"]]
df["sample_85_05"] = [sample_size(size, 85.0, 5.0) for size in df["count"]]
df["sample_85_10"] = [sample_size(size, 85.0, 10.0) for size in df["count"]]
df["sample_90_05"] = [sample_size(size, 90.0, 5.0) for size in df["count"]]
df["sample_90_10"] = [sample_size(size, 90.0, 10.0) for size in df["count"]]
df["sample_95_05"] = [sample_size(size, 95.0, 5.0) for size in df["count"]]
df["sample_95_10"] = [sample_size(size, 95.0, 10.0) for size in df["count"]]
df.to_csv("res_sample_sizes.csv", index=False)
