from typing import List
import matplotlib.pyplot as plt
import os
import scipy

logs_path = "../../logs"


def process_file(file_name: str) -> List[float]:
    """
    This function assumes that the content of the file is of the form
    Epoch [1/4], Step [13/4624], Loss: 4.9277, Perplexity: 138.0610
    Args:
        file_name: the name of the file that wants to be processed
    Returns:
        List[float]: a list with all the train losses
    """
    print(f"Processing file {file_name}...")

    losses = []
    
    with open(file_name) as file:
        for line in file:
            loss = float(line.split(",")[2].split(":")[1])
            losses.append(loss)
    
    
    return losses



fig = plt.subplot()
for file in os.listdir(logs_path):
    if file.endswith(".log"):
        loss = process_file(f"{logs_path}/{file}")
        smoothed_loss = scipy.signal.savgol_filter(loss, 100, 5)
        fig.plot(smoothed_loss, label=file.split(".")[0])
        
        
plt.title("Training loss evolution for the different methods", fontsize=14, fontweight="bold")
plt.grid()
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
