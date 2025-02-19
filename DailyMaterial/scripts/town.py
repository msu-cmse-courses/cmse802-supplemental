import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.spatial import distance

class TownOptimization:
    def __init__(self, name = "East Lansing", n_houses=100, x_dim=10, y_dim=10, method='halton', distance_metric='L1'):
        """
        Initialize a town with houses placed using various distribution methods.
        
        Parameters:
            n_houses: Number of houses to place
            x_dim, y_dim: Town dimensions
            method: House placement method - 'uniform', 'halton', 'sobol', 'grid'
        """
        self.name = name
        self.n_houses = n_houses
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.houses = self._generate_houses(method)
        self.existing_store = None
        self.distance_metric = distance_metric

    def _generate_houses(self, method = 'halton'):
        """
        Generate house locations using different distribution methods.
        Each method has different properties that affect how houses are distributed:
        - Uniform: Completely random, may create clusters
        - Halton: Low-discrepancy sequence, more evenly spaced
        - Sobol: Another low-discrepancy sequence, very uniform coverage
        - Grid: Regular grid with random occupation, like city blocks
        """
        if method == 'uniform':
            # Simple random distribution
            rng = np.random.default_rng()
            points = rng.random((self.n_houses, 2))
            
        elif method == 'halton':
            # Halton sequence - deterministic, low-discrepancy
            sampler = qmc.Halton(d=2, seed=42)
            points = sampler.random(self.n_houses)
            
        elif method == 'sobol':
            # Sobol sequence - another low-discrepancy sequence
            sampler = qmc.Sobol(d=2, seed=42)
            points = sampler.random(self.n_houses)
            
        elif method == 'grid':
            # Grid-based placement with random occupation
            # Similar to your original code
            x = []
            y = []
            for i in range(self.x_dim):
                for j in range(self.y_dim):
                    if np.random.random() < (self.n_houses / (self.x_dim * self.y_dim)):
                        x.append(i)
                        y.append(j)
            points = np.column_stack((np.array(x)/self.x_dim, np.array(y)/self.y_dim))
            
        else:
            raise ValueError(f"Unknown distribution method: {method}")
            
        # Scale points to town dimensions
        return points * [self.x_dim, self.y_dim]

    def plot_town(self, store_location=None, title=None):
        """Visualize town layout with optional store location"""
        plt.figure(figsize=(10, 10))
        plt.scatter(self.houses[:, 0], self.houses[:, 1], 
                   c='blue', alpha=0.5, label='Houses')
        
        if self.existing_store is not None:
            plt.scatter(self.existing_store[0], self.existing_store[1], 
                       color='red', s=200, marker='*', 
                       label='Existing Store')
            
        if store_location is not None:
            plt.scatter(store_location[0], store_location[1], 
                       color='green', s=200, marker='*', 
                       label='New Store')
            
        if title:
            plt.title(title)
        else:
            plt.title(self.name)
        plt.xlabel("Distance (units)")
        plt.ylabel("Distance (units)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def average_distance_to_store(self, store_location):
        """
        Calculate average distance from houses to store location.
        
        This method supports two distance metrics:
        1. Manhattan Distance (L1 norm):
           - Measures distance as |x1 - x2| + |y1 - y2|
           - Represents travel along city blocks
           - More realistic for grid-like street layouts
        
        2. Euclidean Distance (L2 norm):
           - Measures straight-line distance √((x1-x2)² + (y1-y2)²)
           - Represents "as the crow flies" distance
           - Useful when direct paths are possible
        
        Parameters:
            store_location: np.array([x, y]) - Coordinates of the store
            
        Returns:
            float: Average distance from all houses to the store
        """
        # Calculate coordinate differences
        x_diff = self.houses[:, 0] - store_location[0]
        y_diff = self.houses[:, 1] - store_location[1]
        
        if self.distance_metric == 'L1':
            # Manhattan distance: sum of absolute differences
            distances = np.abs(x_diff) + np.abs(y_diff)
        elif self.distance_metric == 'L2':
            # Euclidean distance: square root of sum of squares
            distances = np.sqrt(x_diff**2 + y_diff**2)
        else:
            raise ValueError("Distance metric must be either 'L1' or 'L2'")
            
        return np.mean(distances)

    def distance_to_store(self, house, store_location):
        """
        Calculate distance from a single house to the store.
        
        This method supports two distance metrics:
        1. Manhattan Distance (L1 norm)
        2. Euclidean Distance (L2 norm)
        
        Parameters:
            house: np.array([x, y]) - Coordinates of the house
            store_location: np.array([x, y]) - Coordinates of the store
            
        Returns:
            float: Distance from the house to the store
        """
        x_diff = house[0] - store_location[0]
        y_diff = house[1] - store_location[1]
        
        if self.distance_metric == 'L1':
            return np.abs(x_diff) + np.abs(y_diff)
        elif self.distance_metric == 'L2':
            return np.sqrt(x_diff**2 + y_diff**2)
        else:
            raise ValueError("Distance metric must be either 'L1' or 'L2'")
        

    def visualize_distances(self, store_location):
        """
        Visualize how distances are calculated from store to houses.
        
        This visualization helps students understand the difference between
        Manhattan and Euclidean distances by showing actual paths and
        distance distributions.
        """
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Show paths from store to selected houses
        plt.subplot(1, 2, 1)
        plt.scatter(self.houses[:, 0], self.houses[:, 1], 
                   c='blue', alpha=0.5, label='Houses')
        plt.scatter(store_location[0], store_location[1], 
                   color='red', s=200, marker='*', label='Store')
        
        # Show paths for a few example houses
        for house in self.houses[:5]:
            if self.distance_metric == 'L1':
                # Show Manhattan path
                plt.plot([store_location[0], house[0], house[0]], 
                        [store_location[1], store_location[1], house[1]], 
                        'k--', alpha=0.3)
            else:
                # Show Euclidean path
                plt.plot([store_location[0], house[0]], 
                        [store_location[1], house[1]], 
                        'k--', alpha=0.3)
        
        plt.title(f"Paths from Store to Houses ({self.distance_metric} distance)")
        plt.grid(True)
        
        # Plot 2: Distance distribution
        plt.subplot(1, 2, 2)
        if self.distance_metric == 'L1':
            distances = np.abs(self.houses[:, 0] - store_location[0]) + \
                       np.abs(self.houses[:, 1] - store_location[1])
        else:
            distances = np.sqrt(
                (self.houses[:, 0] - store_location[0])**2 + 
                (self.houses[:, 1] - store_location[1])**2
            )
            
        plt.hist(distances, bins=20)
        plt.title(f"Distribution of {self.distance_metric} Distances")
        plt.xlabel("Distance")
        plt.ylabel("Number of Houses")
        
        plt.tight_layout()
        plt.show()