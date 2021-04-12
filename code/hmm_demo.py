from dmarsenal.series.HMM import HMM

state_sequence = "阴-阴-晴-晴-阴-雨-阴-雨-晴-雨-阴-阴-雨-雨-晴-晴-阴-阴-阴"
obs_sequence = "打-不-不-打-不-打-不-不-不-打-打-不-打-打-不-不-不-打-不"

state_seq = state_sequence.split(sep='-')
obs_seq = obs_sequence.split(sep='-')

print("Train state sequence:",state_seq)
print("Train observation sequence:",obs_seq)

S = list(set(state_seq))
O = list(set(obs_seq))

model = HMM(N=len(S), M=len(O), list_states=S, list_obs=O, random_state=None)

model.train(obs_seq, 100)
print("Predict probability:",model.cal_prob(obs_seq))
print("Predicted state sequence:",model.predict(obs_seq))
