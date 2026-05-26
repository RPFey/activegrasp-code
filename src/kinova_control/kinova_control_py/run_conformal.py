import numpy as np
import os
import pickle
import tqdm 
import argparse
from matplotlib import pyplot as plt
import glob
import re
from enum import Enum

class GraspResult(Enum):
    SUCCESS = 1
    NO_VALID_GRASP = 2
    IK_FAIL = 3
    BAD_POSE = 4
    DROP = 5

def run_conformal(root_dir, seed = -1, episode = -1):
    if seed >= 0 and episode >= 0:
        print(f"Seed: {seed}, Episode: {episode}")
        exps = glob.glob(os.path.join(root_dir, f's{seed}-ep{episode}', '*', '*'))
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
    active = info.split('_')[-1]
    grasp = '_'.join(info.split('_')[:-1])
    print(grasp, active)

    for exp in tqdm.tqdm(exps, desc='Calibration ...'):
        iter_name = '/'.join(exp.split('/')[-3:-1])

        if os.path.exists(os.path.join(exp, 'curobo.log')):
            with open(os.path.join(exp, 'curobo.log'), 'r') as f:
                lines = f.readlines()
                lines.reverse()
                status_found = False
                for line in lines:
                    if line.startswith("Grasp Status:"):
                        status_found = True
                        # Find all numbers inside brackets
                        status = re.findall(r'\d+', line)
                        # Convert to integers
                        status = [int(num) for num in status]
                        if len(status) == 0:
                            invalid.append(iter_name)
                        else:
                            status = status[0]
                            if status == 1:
                                print(line)
                                success.append(iter_name)
                                results.append(2)
                            elif status == 5:
                                drop.append(iter_name)
                                results.append(1)
                            elif status == 4:
                                fail.append(iter_name)
                                results.append(0)
                        break
                if not status_found:
                    unfinished.append(iter_name)
        else:
            unfinished.append(iter_name)
            
    # plt.figure()
    # plt.plot(np.array(results), np.array(scores), 'o')
    # plt.xlabel('Results')
    # plt.ylabel('Scores')
    # plt.title(f"{len(exps)} Trials ; {len(scores)} Valid")
    # plt.savefig('scores.png')

    results = np.array(results)
    print("Success Rate: ", np.sum(results == 2) )
    print("Drop Rate: ", np.sum(results == 1))
    print("Fail Rate: ", np.sum(results == 0))
    print("Invalid: ", len(invalid))
    print("Unfinished: ", len(unfinished))
    print("Compact: ({:d}/{:d}/{:d}/{:d}/{:d})".format(
        np.sum(results == 2), np.sum(results == 1), np.sum(results == 0), len(invalid), len(unfinished)))
    
    pairs = []
    commands = []
    for k in unfinished:
        i = k.split('/')[0]
        seed, ep = i.split('-')
        seed = int(seed[1:])
        ep = int(ep[2:])
        pairs.append((seed, ep))
        
        iteration = k.split('/')[1]
        iter_num = int(iteration[4:])
        
        command = f"sbatch /home/leiboshu/ActiveGrasp/ActiveTouch/slurm/f2a2-ap-single.sh {seed} {ep} {grasp} {active} {iter_num} "
        print(command)
        commands.append(command)
    
    print("Unfinished", pairs)
     
    run_command = input("Run conformal again? (y/n): ")
    if run_command == 'y':
        for command in commands:
            os.system(command)
    
def run_ap(root_dir, seed = -1, episode = -1):
    if seed >= 0 and episode >= 0:
        print(f"Seed: {seed}, Episode: {episode}")
        exps = glob.glob(os.path.join(root_dir, f's{seed}-ep{episode}', '*', '*'))
    else:
        exps = glob.glob(os.path.join(root_dir, 's*-ep*', '*', '*'))
    
    exps.sort()
    scores = []
    unfinished = []

    for exp in tqdm.tqdm(exps, desc='Calibration ...'):
        iter_name = '/'.join(exp.split('/')[-3:-1])

        if os.path.exists(os.path.join(exp, 'curobo.log')):
            with open(os.path.join(exp, 'curobo.log'), 'r') as f:
                lines = f.readlines()
                lines.reverse()
                status_found = False
                for line in lines:
                    if line.startswith("Grasp Results:"):
                        status_found = True
                        # Find all numbers inside brackets
                        success = re.findall(r'\d+', line)
                        # Convert to integers
                        success = [int(num) for num in success]
                        if len(success) == 0:
                            continue
                        AP = sum(success) / len(success)
                        scores.append(AP)   
                        break
                
                if not status_found:
                    unfinished.append(iter_name)
        else:
            unfinished.append(iter_name)

    print("AP: ", np.array(scores).mean())
    print("Unfinished: ", len(unfinished))

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--root_dir', type=str, default='../../data/kinova_control')  
    args.add_argument('--seed', type=int, default=-1)
    args.add_argument('--episode', type=int, default=-1)
    args.add_argument('--ap', action='store_true', default=False)
    args = args.parse_args()
    
    if args.ap:
        run_ap(args.root_dir, args.seed, args.episode)
    else:
        run_conformal(args.root_dir, args.seed, args.episode)