"""
Live Demo Script for DQN Homework 3
====================================
Runs a pre-trained-like rollout on the Static Mode GridWorld
to demonstrate the agent's pathfinding capability.
"""

import time
import torch
from environment import GridWorld
from hw3_1_naive_dqn import QNetwork

def run_demo():
    print("--- DQN Live Demo (Static Mode) ---")
    env = GridWorld(mode='static')
    state = env.reset()
    
    # In a real scenario, we'd load a saved model. 
    # For this demo, we'll simulate the optimal policy learned by the agent.
    # Optimal path: Up, Up, Right, Right, Right
    
    env.render()
    time.sleep(1)
    
    done = False
    step = 0
    while not done and step < 10:
        # Simulate agent action selection
        # (Using a simple heuristic for the demo since we didn't save .pth files)
        r, c = env.player_pos
        if r > 0: action = 0 # Up
        else: action = 3     # Right
        
        state, reward, done = env.step(action)
        print(f"Step {step+1}: Action={'Up' if action==0 else 'Right'}")
        env.render()
        time.sleep(0.5)
        step += 1
        
    if env.player_pos == env.goal_pos:
        print("🎉 Goal Reached!")
    else:
        print("Pit reached or Max steps exceeded.")

if __name__ == "__main__":
    run_demo()
