import matplotlib.pyplot as plt
import json

# Assuming the JSON data is stored in a file named 'training_data.json'
file_path = 'log.json'

# Reading the JSON data from the file
with open(file_path, 'r') as file:
    data = json.load(file)

# Extracting 'steps' and 'loss' values
steps = [item['steps'] for item in data]
losses = [item['loss'] for item in data]

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(steps, losses, marker='o')
plt.title('Training Loss over Steps')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.grid(True)
plt.show()


