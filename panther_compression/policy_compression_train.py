
#!/usr/bin/env python3
##  This file trains and evaluates student policy  


" ********************* Imports ********************* "


import matplotlib.pyplot as plt
import sys
import pathlib
import os
import tempfile
import time
import argparse
import numpy as np
import torch
import random
import subprocess
from colorama import Fore, Style
from imitation.util import util, logger
from imitation.algorithms import bc
from compression.policies.ExpertPolicy import ExpertPolicy
from compression.utils.train import make_simple_dagger_trainer
from compression.utils.eval import evaluate_policy
from IPython.core import ultratb



" ********************* Coloring of the python errors, https://stackoverflow.com/a/52797444/6057617 ********************* "



def printInBoldBlue(data_string):
    print(Style.BRIGHT+Fore.BLUE+data_string+Style.RESET_ALL)
def printInBoldRed(data_string):
    print(Style.BRIGHT+Fore.RED+data_string+Style.RESET_ALL)
def printInBoldGreen(data_string):
    print(Style.BRIGHT+Fore.GREEN+data_string+Style.RESET_ALL)



" ********************* Take string as args and convert it to bool ********************* " 
# See https://stackoverflow.com/a/43357954/6057617



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



" ********************* Preliminary evaluation ********************* "



def preliminary_evaluation(test_venv, expert_policy, LOG_PATH, args, trainer):
    printInBoldBlue("\n----------------------- Preliminary Evaluation: --------------------\n")

    test_venv.env_method("changeConstantObstacleAndGtermPos", gterm_pos=np.array([[6.0],[0.0],[1.0]]), obstacle_pos=np.array([[2.0],[0.0],[1.0]])) 

    #NOTES: args.total_demos_per_round_for_evaluation is the number of trajectories collected in the environment
    #A trajectory is defined as a sequence of steps in the environment (until the environment returns done)
    #Hence, each trajectory usually contains the result of test_len_episode_max_steps timesteps (it may contains less if the environent returned done before) 
    #In other words, episodes in |evaluate_policy() is the number of trajectories
    #                            |the environment is the number of time steps

    ##
    ## Evaluate the reward of the expert
    ##

    expert_stats = evaluate_policy(expert_policy, test_venv, eval_episodes=args.total_demos_per_round_for_evaluation, log_path=LOG_PATH+"/teacher")
    print("[Evaluation] Expert reward: {}, len: {}.\n".format( expert_stats["return_mean"], expert_stats["len_mean"]))

    ##
    ## Evaluate student reward before training,
    ##

    pre_train_stats = evaluate_policy(trainer.policy, test_venv, eval_episodes=args.total_demos_per_round_for_evaluation, log_path=LOG_PATH+"/student_pre_train")
    print("[Evaluation] Student reward: {}, len: {}.".format(pre_train_stats["return_mean"], pre_train_stats["len_mean"]))

    del expert_stats



" ********************* main_train definition ******************* "



def main_train(thread_count, args, gnn_hidden_channels, gnn_num_layers, gnn_num_heads, group, 
               num_linear_layers, linear_hidden_channels, out_channels, num_of_trajs_per_replan,
               batch_size, N_EPOCHS, use_lr_scheduler, lr, evaluation_data_size, weight_prob,
               reuse_latest_policy, use_one_zero_beta, only_collect_data, record_bag, launch_tensorboard,
               log_interval, num_envs, train_evaluation_rate):

    ## To avoid the RuntimeError: CUDA error: out of memory
    time.sleep(thread_count)

    ## Print
    printInBoldBlue("\n----------------------- Input Arguments: -----------------------\n")
    print(f"Trainer:                        {'DAgger' if args.use_dagger else 'BC'}")
    print(f"n_rounds:                       {args.n_rounds}")
    print(f"train_len_episode_max_steps:    {args.train_len_episode_max_steps}")
    print(f"test_len_episode_max_steps:     {args.test_len_episode_max_steps}")
    print(f"use_only_last_coll_ds:          {args.use_only_last_coll_ds}")
    print(f"DAgger rampdown_rounds:         {args.rampdown_rounds}")
    print(f"total_demos_per_round:          {args.total_demos_per_round}")
    print(f"gnn_hidden_channels:            {gnn_hidden_channels}")
    print(f"gnn_num_layers:                 {gnn_num_layers}")
    print(f"gnn_num_heads:                  {gnn_num_heads}")
    print(f"num_linear_layers:              {num_linear_layers}")
    print(f"linear_hidden_channels:         {linear_hidden_channels}")
    print(f"out_channels:                   {out_channels}")
    print(f"num_of_trajs_per_replan:        {num_of_trajs_per_replan}")
    print(f"batch_size:                     {batch_size}")
    print(f"N_EPOCHS:                       {N_EPOCHS}")
    print(f"use_lr_scheduler:               {use_lr_scheduler}")

    ## Params
    DATA_POLICY_PATH = os.path.join(args.policy_dir, str(args.seed))
    EVALUATION_DATA_POLICY_PATH = os.path.join(args.evaluation_data_dir, str(args.seed))
    LOG_PATH = os.path.join(args.log_dir, str(args.seed))
    FINAL_POLICY_NAME = "final_policy.pt"
    ENV_NAME = "my-environment-v1"

    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)

    if not os.path.exists(EVALUATION_DATA_POLICY_PATH):
        os.makedirs(EVALUATION_DATA_POLICY_PATH)

    t0 = time.time()

    ## Seeds
    torch.manual_seed(args.seed+thread_count)
    np.random.seed(args.seed+thread_count)
    random.seed(args.seed+thread_count)

    ## Create and set properties for TRAINING environment:
    printInBoldBlue("\n----------------------- Making Environments: -------------------\n")
    train_venv = util.make_vec_env(env_name=ENV_NAME, n_envs=num_envs, seed=args.seed, parallel=False)#Note that parallel applies to the environment step, not to the expert step
    train_venv.seed(args.seed)
    train_venv.env_method("set_len_ep", (args.train_len_episode_max_steps))
    print("[Train Env] Ep. Len:  {} [steps].".format(train_venv.get_attr("len_episode")))

    for i in range(num_envs):
        train_venv.env_method("setID", i, indices=[i]) 

    if record_bag:
        for i in range(num_envs):
            train_venv.env_method("startRecordBag", indices=[i]) 

    if args.init_eval or args.final_eval or args.eval:
        # Create and set properties for EVALUATION environment
        print("[Test Env] Making test environment...")
        test_venv = util.make_vec_env(env_name=ENV_NAME, n_envs=num_envs, seed=args.seed, parallel=False)#Note that parallel applies to the environment step, not to the expert step
        test_venv.seed(args.seed)
        test_venv.env_method("set_len_ep", (args.test_len_episode_max_steps)) 
        print("[Test Env] Ep. Len:  {} [steps].".format(test_venv.get_attr("len_episode")))

        for i in range(num_envs):
            test_venv.env_method("setID", i, indices=[i]) 

    # Init logging
    tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
    tempdir_path = LOG_PATH#"evals/log_tensorboard"#LOG_PATH#pathlib.Path(tempdir.name)
    print( f"All Tensorboards and logging are being written inside {tempdir_path}/.")
    custom_logger=logger.configure(tempdir_path, format_strs=["log", "csv", "tensorboard"])



    " ********************* Create policies *********************"



    # Create expert policy 
    expert_policy = ExpertPolicy()

    ## Create student policy
    trainer = make_simple_dagger_trainer(tmpdir=DATA_POLICY_PATH, eval_dir=EVALUATION_DATA_POLICY_PATH, venv=train_venv, 
                                            rampdown_rounds=args.rampdown_rounds, custom_logger=custom_logger, lr=lr, 
                                            use_lr_scheduler=use_lr_scheduler, batch_size=batch_size,
                                            evaluation_data_size=evaluation_data_size, weight_prob=weight_prob, 
                                            expert_policy=expert_policy, type_loss=args.type_loss, only_test_loss=args.only_test_loss, 
                                            epsilon_RWTA=args.epsilon_RWTA, reuse_latest_policy=reuse_latest_policy, 
                                            use_one_zero_beta=use_one_zero_beta,
                                            gnn_hidden_channels=gnn_hidden_channels, gnn_num_layers=gnn_num_layers, gnn_num_heads=gnn_num_heads, group=group, 
                                            num_linear_layers=num_linear_layers, linear_hidden_channels=linear_hidden_channels, out_channels=out_channels,
                                            num_of_trajs_per_replan=num_of_trajs_per_replan, train_evaluation_rate=train_evaluation_rate)

    ## Create policy for evaluation data set
    evaluation_trainer = make_simple_dagger_trainer(tmpdir=EVALUATION_DATA_POLICY_PATH, eval_dir=EVALUATION_DATA_POLICY_PATH, venv=train_venv, 
                                                    rampdown_rounds=args.rampdown_rounds, custom_logger=None, lr=lr, use_lr_scheduler=use_lr_scheduler, batch_size=batch_size, 
                                                    evaluation_data_size=evaluation_data_size, weight_prob=weight_prob, expert_policy=expert_policy, 
                                                    type_loss=args.type_loss, only_test_loss=args.only_test_loss, epsilon_RWTA=args.epsilon_RWTA, 
                                                    reuse_latest_policy=reuse_latest_policy, use_one_zero_beta=True,
                                                    gnn_hidden_channels=gnn_hidden_channels, gnn_num_layers=gnn_num_layers, gnn_num_heads=gnn_num_heads, group=group,
                                                    num_linear_layers=num_linear_layers, linear_hidden_channels=linear_hidden_channels, out_channels=out_channels,
                                                    num_of_trajs_per_replan=num_of_trajs_per_replan, train_evaluation_rate=train_evaluation_rate)



    " ********************* Collect evaluation data *********************" 


    # deprecated - right now we split the data into train and eval in bc.py
    if args.evaluation_data_collection:
        printInBoldBlue("\n----------------------- Collecting Evaluation Data: --------------------\n")
        evaluation_policy_path = os.path.join(EVALUATION_DATA_POLICY_PATH, "evaluation_policy.pt") # Where to save curr policy
        evaluation_trainer.train(n_rounds=1, total_demos_per_round=evaluation_data_size, only_collect_data=True, 
                                    bc_train_kwargs=dict(n_epochs=N_EPOCHS, save_full_policy_path=evaluation_policy_path, 
                                                        log_interval=log_interval))



    " ********************* Preliminiary evaluation *********************"



    if args.init_eval:
        preliminary_evaluation(test_venv, expert_policy, LOG_PATH, args, trainer)
        


    " ********************* Training *********************"
    printInBoldBlue("\n----------------------- Training Student: --------------------\n")



    # Launch tensorboard visualization
    if launch_tensorboard:
        os.system("pkill -f tensorboard")
        proc1 = subprocess.Popen(["tensorboard", "--logdir", LOG_PATH, "--bind_all"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    # Train
    if args.train:
        stats = {"training":list(), "eval_no_dist":list()}
        if args.use_dagger:
            assert trainer.round_num == 0
        policy_path = os.path.join(DATA_POLICY_PATH, "intermediate_policy.pt") # Where to save curr policy
        trainer.train(n_rounds=args.n_rounds, total_demos_per_round=args.total_demos_per_round, only_collect_data=only_collect_data, bc_train_kwargs=dict(n_epochs=N_EPOCHS, save_full_policy_path=policy_path, log_interval=log_interval))

        # Store the final policy.
        save_full_policy_path = os.path.join(DATA_POLICY_PATH, FINAL_POLICY_NAME)
        trainer.save_policy(save_full_policy_path)
        print(f"[Trainer] Training completed. Policy saved to: {save_full_policy_path}.")



    " ********************* Evaluation *********************" 
    printInBoldBlue("\n----------------------- Evaluation After Training: --------------------\n")



    if args.final_eval:

        # Evaluate reward of student post-training
        # Note: no disturbance
        post_train_stats = dict()

        # Print
        if args.init_eval: # if there's an initial evaluation, compare the pre and post training rewards

            post_train_stats = evaluate_policy(trainer.policy, test_venv, eval_episodes=args.total_demos_per_round_for_evaluation, log_path=LOG_PATH + "/student_post_train" )
            print("[Complete] Reward: Pre: {}, Post: {}.".format( pre_train_stats["return_mean"], post_train_stats["return_mean"]))

            printInBoldBlue("\n----------------------- Improvement: --------------------\n")

            if(abs(pre_train_stats["return_mean"])>0):
                student_improvement=(post_train_stats["return_mean"]-pre_train_stats["return_mean"])/abs(pre_train_stats["return_mean"])
                if(student_improvement>0):
                    printInBoldGreen(f"Student improvement: {student_improvement*100}%")
                else:
                    printInBoldRed(f"Student improvement: {student_improvement*100}%")
            
            print("[Complete] Episode length: Pre: {}, Post: {}.".format( pre_train_stats["len_mean"], post_train_stats["len_mean"]))

            # Clean up
            del pre_train_stats, post_train_stats

        # Load and evaluate the saved DAgger policy
        load_full_policy_path = os.path.join(DATA_POLICY_PATH, FINAL_POLICY_NAME)
        final_student_policy = bc.reconstruct_policy(load_full_policy_path)
        rwrd = evaluate_policy(final_student_policy, test_venv, eval_episodes=args.total_demos_per_round_for_evaluation, log_path=None)

        # Evaluate the reward of the expert as a sanity check
        expert_reward = evaluate_policy(expert_policy, test_venv, eval_episodes=args.total_demos_per_round_for_evaluation, log_path=None)

        # Print
        printInBoldRed("----------------------- TEST RESULTS: --------------------")
        print("[Test] Student Policy: Avg. Cost: {}, Success Rate: {}".format(-rwrd["return_mean"], rwrd["success_rate"]))
        print("[Test] Expert Policy: Avg. Cost: {}, Success Rate: {}".format(-expert_reward["return_mean"], expert_reward["success_rate"]))

    print("Elapsed time: {}".format(time.time() - t0))



" ********************* Main ********************* "



def main():

    " ********************* Parse args *********************"

    parser = argparse.ArgumentParser()

    ## Housekeeping params

    path = str(pathlib.Path(__file__).parent.absolute())+"/" # path to this file
    parser.add_argument("--home_dir", type=str, default=path)
    parser.add_argument("--log_dir", type=str, default=path+"evals/log_dagger") # usually "log"
    parser.add_argument("--policy_dir", type=str, default=path+"evals/tmp_dagger") # usually "tmp"
    parser.add_argument("--evaluation_data_dir", type=str, default=path+"evals/evalations") # usually "tmp"

    ## Toggles for test

    parser.add_argument("-t", "--use-test-run-params", default=False, type=str2bool)
    parser.add_argument("--only_test_loss", type=str2bool, default=False)
    DEFAULT_N_ROUNDS = 100 if not parser.parse_args().use_test_run_params else 100
    DEFAULT_TOTAL_DEMOS_PER_ROUND = 256*5 if not parser.parse_args().use_test_run_params else 10
    only_collect_data = False # when you want to collect data and not train student

    ## Evaluation params

    parser.add_argument("--total_demos_per_round_for_evaluation", default=100, type=int)
    reset_evaluation_data = True # reset evaluation data
    evaluation_data_size = 100 if not parser.parse_args().use_test_run_params else 10 # evaluation batch size

    ## Dagger params

    use_one_zero_beta = False # use one zero beta in DAagger? if False, it will be LinearBetaSchedule()

    ## Training params

    parser.set_defaults(use_dagger=True) # Default will be to use DAgger
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--rampdown_rounds", default=5, type=int) # learning rate rampdown rounds
    parser.add_argument("--train_len_episode_max_steps", default=100, type=int)
    parser.add_argument("--test_len_episode_max_steps", default=50, type=int)
    parser.add_argument("--n_rounds", default=DEFAULT_N_ROUNDS, type=int)
    parser.add_argument("--total_demos_per_round", default=DEFAULT_TOTAL_DEMOS_PER_ROUND, type=int)
    parser.add_argument("--train", dest='train', action='store_false')
    parser.add_argument("--eval", dest='eval', action='store_false')
    parser.add_argument("--init_eval", dest='init_eval', action='store_true')
    parser.add_argument("--final_eval", dest='final_eval', action='store_true')
    parser.add_argument("--use_only_last_collected_dataset", dest='use_only_last_coll_ds', action='store_true')
    parser.add_argument("--evaluation_data_collection", dest='evaluation_data_collection', action='store_true')
    train_only_from_existing_data = True # when you want to train student only from existing data
    reuse_previous_samples = True # use the existing data?
    reuse_latest_policy = False # reuse the latest_policy?

    ## NN hyperparams

    parser.add_argument("--type_loss", type=str, default="Hung") 
    parser.add_argument("--epsilon_RWTA", type=float, default=0.05)
    batch_size = 128 if not parser.parse_args().use_test_run_params else 5 # batch size
    N_EPOCHS = 50 if not parser.parse_args().use_test_run_params else 1 # epoch size
    use_lr_scheduler = False # use learning rate schedule?
    lr = 1e-3 # constant learning rate (if use_lr_scheduler is False)
    weight_prob = 0.005 # probably not used
    num_envs = 10 if parser.parse_args().use_test_run_params else 16  # number of environments

    ## Data collection params
    record_bag = False # record bags?
    launch_tensorboard = True # use tensorboard?
    verbose_python_errors = False # verbose python errors?
    log_interval = 200 # log stats after every log_interval batches.
    train_evaluation_rate = 1.0 # split the data into train and eval sets (train_evaluation_rate is the percentage of data that goes into the train set)

    ## GNN params

    gnn_hidden_channels = 128 # the size of the hidden layer in GNN
    gnn_num_layers = 2 # the number of GNN layers
    gnn_num_heads = 2 # the number of heads in GNN
    group = 'mean' # 'sum'
    num_linear_layers = 8 # the number of linear layers following GNN
    linear_hidden_channels = 128 # the size of the hidden layer in the linear layers following GNN
    out_channels = 22 # 22 is the number of actions (traj size)
    num_of_trajs_per_replan = 10 # number of trajectories per replan

    ## expose args and params
    args = parser.parse_args()
    
    " ********************* Housekeeping *********************"

    assert args.total_demos_per_round >= batch_size #If not, round_{k+1} will train on the same dataset as round_{k} (until enough rounds are taken to generate a new batch of demos)
    os.system("rm -rf "+args.log_dir) #Delete the logs
    
    # if you only collect data, you don't need to train the student
    if only_collect_data:
        train_only_from_existing_data = False
        launch_tensorboard = False

    # reset stored evaluation data
    if reset_evaluation_data:
        evals_dir=args.evaluation_data_dir+"/2/demos"
        os.system("rm -rf "+evals_dir+"/round*")
        os.system("mkdir -p "+evals_dir+"/round-000")

    # if you train only supervised, you don't need to use DAgger
    if train_only_from_existing_data:
        reuse_previous_samples=True
        only_collect_data=False 
        log_interval=15 
        num_envs=1
        demos_dir=args.policy_dir+"/2/demos/"

        ##
        ## This places all the demos in the round-000 folder
        ##

        os.system("find "+demos_dir+" -type f -print0 | xargs -0 mv -t "+demos_dir)
        os.system("rm -rf "+demos_dir+"/round*")
        os.system("mkdir "+demos_dir+"/round-000")
        os.system("mv "+demos_dir+"/*.npz "+demos_dir+"/round-000")

        ##
        ## Find max round in demos folder
        ##

        max_round = max([int(s.replace("round-", "")) for s in os.listdir(demos_dir)])
        args.n_rounds=max_round+1; #It will use the demonstrations of these folders
        args.total_demos_per_round=0

    if args.only_test_loss:
        batch_size=1; N_EPOCHS=1; log_interval=1

    if args.train_len_episode_max_steps> 1 and only_collect_data:
        printInBoldRed("Note that DAgger will not be used (since we are only collecting data)")

    if args.only_test_loss==False and reuse_latest_policy==False:
        os.system("find "+args.policy_dir+" -type f -name '*.pt' -delete") #Delete the policies

    if reuse_previous_samples==False:
        os.system("rm -rf "+args.policy_dir) #Delete the demos

    if record_bag:
        os.system("rm training*.bag")

    mode='Plain' if not verbose_python_errors else 'Verbose'

    ## Coloring of the python errors, https://stackoverflow.com/a/52797444/6057617
    sys.excepthook = ultratb.FormattedTB(mode=mode, color_scheme='Linux', call_pdb=False)
    
    " ********************* main_train ********************* "

    thread_count = 0
    main_train(thread_count, args, gnn_hidden_channels, gnn_num_layers, gnn_num_heads, group,
               num_linear_layers, linear_hidden_channels, out_channels, num_of_trajs_per_replan,
               batch_size, N_EPOCHS, use_lr_scheduler, lr, evaluation_data_size, weight_prob,
               reuse_latest_policy, use_one_zero_beta, only_collect_data, record_bag, launch_tensorboard,
               log_interval, num_envs, train_evaluation_rate)

if __name__ == "__main__":
    main()
