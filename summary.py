import argparse
import glob
import os
import pickle
import re
import shutil
from enum import Enum

import numpy as np
import tqdm
from matplotlib import pyplot as plt

TOTAL_SEEDS, TOTAL_EPISODES, TOTAL_ITERATIONS = 20, 5, 4

class GraspResult(Enum):
    SUCCESS = 1
    NO_VALID_GRASP = 2
    IK_FAIL = 3
    BAD_POSE = 4
    DROP = 5

class ResultSummary:
    def __init__(self, root_dir, seed = -1, episode = -1):
        if seed >= 0 and episode >= 0:
            print(f"Seed: {seed}, Episode: {episode}")
            exps = glob.glob(os.path.join(root_dir, f's{seed}-ep{episode}', '*', '*'))
        elif seed >= 0:
            print(f"Seed: {seed}")
            exps = glob.glob(os.path.join(root_dir, f's{seed}-ep*', '*', '*'))
        else:
            exps = glob.glob(os.path.join(root_dir, 's*-ep*', '*', '*'))
        
        exps.sort()
        scores = []
        results = []
        success = []
        drop = []
        fail = []
        invalid = []
        unfinished = []
        
        exp = exps[0]
        info = exp.split('/')[-4]
        active = info.split('_')[-3]
        grasp = '_'.join(info.split('_')[:-3])
        print(grasp, active)
        
        # TODO Change Total Seeds, Episodes, Iterations here
        unfinished_arr = np.ones((TOTAL_SEEDS, TOTAL_EPISODES, TOTAL_ITERATIONS), dtype=np.uint8)
        unfinished_dir = []

        for exp in tqdm.tqdm(exps, desc='Calibration ...'):
            # iter_name = exp.split('/')[-2]  
            ep_name = exp.split('/')[-3]
            iter_name = '/'.join([exp.split('/')[-3], exp.split('/')[-2]])

            if os.path.exists(os.path.join(exp, 'curobo.log')):
                with open(os.path.join(exp, 'curobo.log'), 'r') as f:
                    lines = f.readlines()
                    # lines.reverse()
                    status_found = False
                    episode_grasp_score = -1
                    
                    for line in lines:
                        if line.startswith("Grasp Phase done"):
                            status_found = True
                            
                            # set flags
                            seed_eps = exp.split('/')[-3].split('-')
                            seed = int(seed_eps[0][1:])
                            epi = int(seed_eps[1][2:]) - 1
                            iters = int(exp.split('/')[-2][4:])
                            
                            if unfinished_arr[seed, epi, iters] == 0:
                                print(f"{exp=}, {ep_name} Finished, but already marked as such")
                                shutil.rmtree(exp, ignore_errors=True)
                                break
                            
                            unfinished_arr[seed, epi, iters] = 0 
            
                            if re.search(r"\bSUCCESS\b", line):
                                success.append(iter_name)
                                results.append(2)
                                scores.append(episode_grasp_score)
                            elif re.search(r"\bDROP\b", line):
                                drop.append(iter_name)
                                results.append(1)
                                scores.append(episode_grasp_score)
                                print(f"{exp=}, {ep_name} DROP")
                            elif re.search(r"\bBAD_POSE\b", line):
                                fail.append(iter_name)
                                results.append(0)
                                scores.append(episode_grasp_score)
                                print(f"{exp=}, {ep_name} BAD_POSE")
                            elif re.search(r"\bIK_FAIL\b", line):
                                invalid.append(iter_name)
                                print(f"{exp=}, {ep_name} IK_FAIL")
                            elif re.search(r"\bNO_VALID_GRASP\b", line):
                                invalid.append(iter_name)
                                print(f"{exp=}, {ep_name} NO_VALID_GRASP")
                                    
                        elif line.startswith("Select Pose with Score:"):
                           episode_grasp_score = float(line.split(':')[-1])
                    
                    if not status_found:
                        unfinished.append(iter_name)
                        unfinished_dir.append(exp)
                        print(f"{exp=}, {ep_name} UNFINISHED")
            else:
                print(f"{exp=}, {ep_name} UNFINISHED")
                unfinished.append(iter_name)
                unfinished_dir.append(exp)
        
        clean_command = input("Clean up finished experiments? (y/n): ")
        if clean_command == 'y':
            for exp in unfinished_dir:
                shutil.rmtree(exp, ignore_errors=True)
                
        # plt.figure()
        # plt.plot(np.array(results), np.array(scores), 'o')
        # plt.xlabel('Results')
        # plt.ylabel('Scores')
        # plt.title(f"{len(exps)} Trials ; {len(scores)} Valid")
        # plt.savefig('scores.png')
        results = np.array(results)

        nbins = 15
        scores = np.array(scores)
        thresholds = np.linspace(0.0, 1, nbins + 1)
        ece = 0
        srs = []
        for i in range(nbins):
            mask = (scores >= thresholds[i]) & (scores < thresholds[i + 1])
            mask_result = results[mask]
            sr = np.sum(mask_result == 2) / len(mask_result) if len(mask_result) > 0 else 0
            print(f"Score Range: [{thresholds[i]:.2f}, {thresholds[i+1]:.2f}) -> Success Rate: {sr:.2f} ({len(mask_result)})")
            calibration_error = abs(sr - (thresholds[i] + thresholds[i + 1]) / 2)
            srs.append(sr)
            ece += calibration_error * len(mask_result) / len(scores) if len(scores) > 0 else 0
        print(f"Expected Calibration Error (ECE): {ece:.4f}")
        print(srs)

        print("Success Rate: ", np.sum(results == 2) )
        print("Drop Rate: ", np.sum(results == 1))
        print("Fail Rate: ", np.sum(results == 0))
        print("Invalid: ", len(invalid))
        print("Unfinished: ", len(unfinished))
        print("Compact: ({:d}/{:d}/{:d}/{:d}/{:d})".format(
            np.sum(results == 2), np.sum(results == 1), np.sum(results == 0), len(invalid), len(unfinished)))
        print(f"Success rate: {np.sum(results == 2) / (len(exps) - len(unfinished)):.2f}")
        
        # print("Success: ", success)
        # print("Drop: ", drop)
        # print("Fail: ", fail)
        # print("Invalid: ", invalid)
        
        unfinished_seed, unfinished_epi, unfinished_iter = np.where(unfinished_arr == 1)
        
        pairs = []
        commands = []
        for seed, ep, iter_num in zip(unfinished_seed, unfinished_epi, unfinished_iter):
            # i = k.split('/')[0]
            # print(i)
            # seed, ep = i.split('-')
            # seed = int(seed[1:])
            # ep = int(ep[2:])
            # iteration = k.split('/')[1]
            # iter_num = int(iteration[4:])
            
            ep_ = ep + 1
            pairs.append((seed, ep_, iter_num))
            command = f"sbatch /home/leiboshu/ActiveGrasp/ActiveTouch/slurm/f2a2-single-lambda.sh {seed} {ep_} {grasp} {active} {iter_num} "
            print(command)
            commands.append(command)
        
        print("Unfinished", pairs)
        
        run_command = input("Run conformal again? (y/n): ")
        if run_command == 'y':
            for command in commands:
                os.system(command)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--root_dir', type=str, default='../../data/kinova_control')  
    args.add_argument('--seed', type=int, default=-1)
    args.add_argument('--episode', type=int, default=-1)
    args = args.parse_args()
    
    calibration = ConformalCalibration(args.root_dir, args.seed, args.episode)
