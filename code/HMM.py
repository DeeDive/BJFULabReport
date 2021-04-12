import numpy as np
import pprint
from dmarsenal.moniter.ConvergenceMonitor import ConvergenceMonitor

class HMM(object):
    def __init__(self, N, M,list_states,list_obervations,random_state=None):
        """HMM model with N states and M observations,
        The model is trained with Baum-Welch Algorothm.

        :param N: number of states
        :param M: number of observations
        :param random_state: seed
        """
        self.N = N
        self.O = None
        self.T = None

        # if random_state:
            # np.random.seed(random_state)
        # random initialization
        self.A = np.random.rand(N, N)
        # 目前是 A_{N_s_{t} x N_s_{t+1}}
        # 为了向量化的计算，这里也可以是A_{N_s_{t+1} x N_s_t}
        # 如果是后者，那么和书上定义稍微有点不同，是其转置
        self.A /= np.sum(self.A, axis=-1).reshape(-1, 1)
        self.B = np.random.rand(N, M)
        self.B /= np.sum(self.B, axis=-1).reshape(-1, 1)
        self.pi = np.ones(N) / N

        self._monitor = None

        self.list_states = list_states
        self.list_obervations = list_obervations
        self.dict_state_to_idx = {state:i for i,state in enumerate(list_states)}
        self.dict_observation_to_idx = {obs:i for i,obs in enumerate(list_obervations)}

        # pprint.pprint(self.A)
        # pprint.pprint(self.B)
        # pprint.pprint(self.pi)

    def train(self, O, max_iters):
        O = self.convert_obs(O)
        self.O = np.array(O)
        self.T = self.O.shape[0]
        self._monitor = ConvergenceMonitor(tol=1E-8, n_iter=max_iters, verbose=False)

        # print(self.forward(),self.backward())
        it = 0
        while True:
            it += 1

            # execution as a side effect of assert
            assert np.isclose(self.forward(), self.backward()) \
                   or print(self.forward(), self.backward())
            prob = self.forward()
            if (it) %5 ==0:
                print("Iteration", it,":",prob)
            self._monitor.report(prob)

            if self._monitor.converged:
                break

            gamma = self.cal_gamma()
            ksi = self.cal_ksi()
            self.A = (np.sum(ksi,axis=0)-ksi[self.T-1])\
                     /((np.sum(gamma,axis=0)-gamma[self.T-1]).reshape(-1,1))
            # update those emission values that are in the observation sequence
            obs = np.unique(self.O)
            for o in obs:
                # print(self.O==o,gamma.shape,np.where(self.O==o))

                # print(np.sum(gamma[np.where(self.O==o)],axis=0).shape)
                # print(np.sum(gamma,axis=0).shape)
                self.B[:,o] = np.sum(gamma[np.where(self.O==o)],axis=0,)\
                              /np.sum(gamma,axis=0)

            self.pi = gamma[0,:]

    def forward(self):
        """Forward algorithm

        :return:
        """
        self.alpha = np.empty((self.T, self.N))

        # boundary value
        self.alpha[0, :] = self.pi * self.B[:, self.O[0]]

        # iterative solving
        for t in range(1,self.T):
            self.alpha[t,:] = self.alpha[t-1,:].reshape(1,-1)\
                .dot(self.A)*self.B[:,self.O[t]]

        return np.sum(self.alpha[self.T-1,:])

    def backward(self):
        self.beta = np.empty((self.T, self.N))

        # boundary value
        self.beta[self.T-1,:]=1

        # iterative solving
        for t in reversed(range(0,self.T-1)):
            self.beta[t,:] = np.sum(np.dot(self.A,(self.B[:,self.O[t+1]]
                                                   *self.beta[t+1,:])
                                           .reshape(-1,1)),axis=-1)

        return np.sum(self.pi*self.B[:,self.O[0]]*self.beta[0,:])

    def cal_gamma(self):
        """
        gamma = (gamma_t(s_t))_{T x N}
        :return:
        """
        gamma_tilda = np.empty((self.T,self.N))
        gamma_tilda = self.alpha*self.beta
        gamma = gamma_tilda/(np.sum(gamma_tilda,axis=-1).reshape(-1,1))

        return gamma

    def cal_ksi(self):
        """
        ksi = (ksi_t(s_i,s_j))_{TxNxN}
        :return:
        """
        ksi = np.zeros((self.T,self.N,self.N))
        for t in range(self.T-1):
            ksi[t] = self.alpha[t,:].reshape(-1,1)*self.A\
                     *(self.B[:,self.O[t+1]].reshape(1,-1))\
                     *(self.beta[t+1,:].reshape(1,-1))
            ksi[t] /= np.sum(ksi[t])

        return ksi

    def cal_prob(self,O):
        O = self.convert_obs(O)
        self.O = np.array(O)
        self.T = self.O.shape[0]
        return self.forward()

    def predict(self, O):
        O = self.convert_obs(O)
        O = np.array(O)
        T = O.shape[0]

        # init
        delta = np.zeros((T,self.N))
        phi = np.zeros((T,self.N),dtype=np.int32)
        delta[0,:] = self.pi*self.B[:,O[0]]

        for t in range(1,T):
            tmp = delta[t-1,:].reshape(-1,1)*self.A
            delta[t] = np.max(tmp,axis=0)*self.B[:,O[t]]
            phi[t] = np.argmax(tmp,axis=0)

        P_star = np.max(delta[T-1,:])

        path = [np.argmax(delta[T-1,:])]
        for t in reversed(range(T-1)):
            # print("PATH",path[-1])
            path.append(phi[t,path[-1]])

        return P_star,self.to_outer_state_seq(path)


    def to_outer_state(self,idx):
        return self.list_states[idx]
    def to_outer_obervation(self, idx):
        return self.list_obervations[idx]

    def obervation_to_idx(self, obs):
        return self.dict_observation_to_idx[obs]
    def state_to_idx(self, state):
        return self.dict_state_to_idx[state]

    def convert_obs(self,O):
        return [self.obervation_to_idx(o) for o in O]
    def to_outer_state_seq(self,seq):
        return [self.to_outer_state(s) for s in seq]


# if __name__ == '__main__':
#     model = HMM(3, 2)
#
#     model.train([1,0,0,1,0,1,0,0,0,1,1,0,1,1,0,0,0,1,0],100)
#     print("A:")
#     pprint.pprint(model.A)
#     print("B:")
#     pprint.pprint(model.B)
#     print("pi:")
#     pprint.pprint(model.pi)
#
#     print(model.cal_prob([1,0,0,1,0,1,0,0,0,1,1,0,1,1,0,0,0,1,0]))
