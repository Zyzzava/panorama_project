import matplotlib.pyplot as plt

def read_matches(file_path):
    distances = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) > 0:
                distances.append(float(parts[-1]))  # Assuming the distance is the last element in each line
    return distances

def plot_histogram(distances, bins=32):
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=bins, color='orange', edgecolor='black')
    plt.title('Histogram of Match Distances')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('histogram.png')  # Save the histogram as an image
    plt.show()

if __name__ == "__main__":
    distances = read_matches('results/matches.txt')
    plot_histogram(distances)