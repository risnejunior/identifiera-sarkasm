import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from common_funs import DB_Handler
from config import Config
"""
Requires matplot lib
"""
gang_a = "hells-angels" #balanced, tags
gang_b = "bandidos" # freq, tags
gang_a2 = "hells-angels-notags" # freq, tags
gang_b2 = "bandidos-notags" # freq, tags
gang_c = "red-devils" #balanced, tags #12 runs

# get human agreement
"""
SELECT a.question_id, count(a.question_id), SUM(a.answer), s.sample_text
FROM `answers` a
LEFT JOIN samples s
ON a.question_id = s.id
GROUP BY question_id 
HAVING count(question_id) = 2
ORDER BY `SUM(a.answer)`  DESC
LIMIT 400
"""

# get agreement between gangs
"""
SELECT SUM(agreement), COUNT() tot
FROM
(
SELECT  a.sample_id as sid, a.correct, b.correct, a.correct = b.correct as agreement
FROM troublemakers a
INNER JOIN troublemakers b
ON a.sample_id = b.sample_id
WHERE a.gang = 'hells-angels-notags'
AND b.gang = 'bandidos-notags'
GROUP BY sid
)


"""
cfg = Config()
db = DB_Handler(cfg.sqlite_file)
rows = db.getRows("troublemakers", gang = 'hells-angels')

sample_count = len(rows)
tally = {}

for row in rows:
	trouble = round(row['trouble'], 1)
	if trouble in tally:
		tally[trouble] += 1
	else: 
		tally[trouble] = 1
print(sample_count)

plt.axis([0, 1, 0, sample_count])
plt.ion()

for trouble_group, count in tally.items():
    plt.scatter(trouble_group, count)
    plt.pause(0.05)

while True:
    plt.pause(0.05)