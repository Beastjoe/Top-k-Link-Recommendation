import numpy as np

person_set = dict()
cnt = 0
with open("wiki.txt", "r") as f:
    line = f.readline()
    while line:
        if line != "\n":
            strs = line.split("\t")
            if strs[0] == 'U' and strs[2] not in person_set:
                person_set[strs[2]] = cnt
                cnt += 1
            elif strs[0] == 'V' and strs[4] not in person_set:
                person_set[strs[4]] = cnt
                cnt += 1
        line = f.readline()

# Adjacency matrix: voter->candidate
X = np.zeros([cnt, cnt])
with open("wiki.txt", "r") as f:
    line = f.readline()
    while line:
        if line != "\n":
            strs = line.split("\t")
            if strs[0] == 'U':
                curr_candidate = strs[2]
            elif strs[0] == 'V':
                candidate_idx = person_set[curr_candidate]
                voter_idx = person_set[strs[4]]
                X[voter_idx][candidate_idx] = 1
        line = f.readline()
np.save("wiki", X)
