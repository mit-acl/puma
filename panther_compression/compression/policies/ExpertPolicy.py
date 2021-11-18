import sys
import numpy as np
import copy
from random import random
from compression.utils.other import ActionManager, ObservationManager, getPANTHERparamsAsCppStruct
from colorama import init, Fore, Back, Style
import py_panther

class ExpertPolicy(object):

    def __init__(self):
        self.am=ActionManager();
        self.om=ObservationManager();

        self.action_shape=self.am.getActionShape();
        self.observation_shape=self.om.getObservationShape();

        self.par=getPANTHERparamsAsCppStruct();

        self.my_SolverIpopt=py_panther.SolverIpopt(self.par);


        self.reset()

    def printwithName(self,data):
        name=Style.BRIGHT+Fore.BLUE+"[Exp]"+Style.RESET_ALL
        print(name+data)

    def reset(self):
        pass

        #From https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py#L41
        # In the case of policies, the prediction is an action.
        # In the case of critics, it is the estimated value of the observation.
    def predict(self, obs_n, deterministic=True):
        obs_n=obs_n.reshape(self.observation_shape) #Not sure why this is needed
        assert obs_n.shape==self.observation_shape, self.name+f"ERROR: obs.shape={obs_n.shape} but self.observation_shape={self.observation_shape}"
        
        obs=self.om.denormalizeObservation(obs_n);

        # self.printwithName(f"Got obs={obs}")
        # self.printwithName(f"Got obs shape={obs.shape}")

        # self.om.printObs(obs)

        # self.printwithName("Calling optimizer")

        # ## Call the optimization
        # init_state=py_panther.state(); #This is initialized as zero. This is A
        # final_state=py_panther.state();#This is initialized as zero. This is G
        # final_state.pos=self.om.getf_g(obs);

        # total_time=self.par.factor_alloc*py_panther.getMinTimeDoubleIntegrator3DFromState(init_state, final_state, self.par.v_max, self.par.a_max)

        # self.my_SolverIpopt.setInitStateFinalStateInitTFinalT(init_state, final_state, 0.0, total_time);
        # self.my_SolverIpopt.setFocusOnObstacle(True);
        # self.my_SolverIpopt.setObstaclesForOpt(self.om.getObstacles(obs));

        # self.my_SolverIpopt.optimize();
        # #### End of call the optimization
        # self.printwithName("===================================================")





        
        Q=random(); #This is the reward I think

        # action = self.am.getRandomAction()
        # action = self.am.getDummyOptimalAction()

        # self.printwithName(f" Returned action={action}")
        # self.printwithName(f"Returned action shape={action.shape}")

        # action=self.am.normalizeAction(action)

        action=self.am.getDummyOptimalNormalizedAction()

        assert action.shape==self.action_shape

        assert not np.isnan(np.sum(action)), f"trying to output nan"

        return action, {"Q": Q}
