import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from environment import GridWorld
import os

def generate_gif():
    env = GridWorld(mode='static')
    state = env.reset()
    
    # Define optimal steps
    # (2,0) -> (1,0) -> (0,0) -> (0,1) -> (0,2) -> (0,3)
    actions = [0, 0, 3, 3, 3] 
    
    frames = []
    # Capture initial state
    frames.append(env._get_state().reshape(3, 4))
    
    for a in actions:
        env.step(a)
        frames.append(env._get_state().reshape(3, 4))
        
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    
    def update(i):
        ax.clear()
        data = frames[i]
        
        # Enhanced visualization colors
        # Wall=-1 (dark gray), Pit=-0.5 (crimson), Empty=0 (white), Goal=0.5 (gold), Player=1 (lime)
        im_data = np.zeros((3, 4, 3))
        for r in range(3):
            for c in range(4):
                val = data[r, c]
                if val == -1:    color = [0.15, 0.15, 0.15] # Wall
                elif val == -0.5: color = [0.86, 0.08, 0.24] # Pit
                elif val == 0.5:  color = [1.0, 0.84, 0.0]  # Goal
                elif val == 1.0:  color = [0.19, 0.8, 0.19] # Player
                else:            color = [0.95, 0.95, 0.95] # Empty
                im_data[r, c] = color
                
        ax.imshow(im_data)
        
        # Add grid lines
        ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        
        ax.set_title(f"DQN Intelligence in Action - Step {i}", fontsize=14, fontweight='bold', pad=20)
        
        # Add high-contrast text labels
        for r in range(3):
            for c in range(4):
                val = data[r, c]
                label = ""
                t_color = "white"
                if val == -1: label = "WALL"
                elif val == -0.5: label = "PIT"
                elif val == 0.5: label = "GOAL"
                elif val == 1.0: label = "AGENT"
                else: t_color = "black"
                
                if label:
                    ax.text(c, r, label, ha='center', va='center', 
                            color=t_color, fontsize=10, fontweight='black')

    ani = FuncAnimation(fig, update, frames=len(frames), repeat=True)
    
    output_path = "demo.gif"
    writer = PillowWriter(fps=2)
    ani.save(output_path, writer=writer)
    plt.close()
    print(f"GIF saved to {output_path}")

if __name__ == "__main__":
    generate_gif()
