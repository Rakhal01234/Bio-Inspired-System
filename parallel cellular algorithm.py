mport numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

class ParallelEdgeDetector:
    def __init__(self, image_size=500, num_processes=None):
        """
        Initialize parallel edge detector
        
        Args:
            image_size (int): Size of synthetic image
            num_processes (int): Number of parallel processes
        """
        self.image_size = image_size
        # Use all available CPU cores if not specified
        self.num_processes = num_processes or mp.cpu_count()
        
        # Create synthetic image
        self.image = self.create_synthetic_image()
    
    def create_synthetic_image(self):
        """
        Create a complex synthetic image with multiple features
        """
        image = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        # Multiple geometric shapes and gradients
        
        # Gradient background
        for x in range(self.image_size):
            for y in range(self.image_size):
                image[x, y] = (x + y) / (2 * self.image_size) * 255
        
        # Circles
        for _ in range(5):
            center_x = np.random.randint(0, self.image_size)
            center_y = np.random.randint(0, self.image_size)
            radius = np.random.randint(20, 50)
            
            for x in range(self.image_size):
                for y in range(self.image_size):
                    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if radius - 5 < distance < radius + 5:
                        image[x, y] = 255
        
        # Rectangles
        for _ in range(3):
            x1 = np.random.randint(0, self.image_size - 50)
            y1 = np.random.randint(0, self.image_size - 50)
            image[x1:x1+50, y1:y1+50] = 255
        
        return image
    
    def parallel_edge_detection_worker(self, chunk_start, chunk_end):
        """
        Parallel worker for edge detection
        
        Args:
            chunk_start (int): Starting row of image chunk
            chunk_end (int): Ending row of image chunk
        
        Returns:
            numpy.ndarray: Detected edges for chunk
        """
        # Cellular Automata Edge Detection Kernel
        kernel = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ])
        
        # Pad the chunk with border pixels
        padded_chunk = np.pad(
            self.image[chunk_start:chunk_end], 
            pad_width=1, 
            mode='edge'
        )
        
        # Initialize edge chunk
        edges_chunk = np.zeros_like(
            self.image[chunk_start:chunk_end], 
            dtype=np.float32
        )
        
        # Cellular Automata Edge Detection
        for x in range(1, padded_chunk.shape[0] - 1):
            for y in range(1, padded_chunk.shape[1] - 1):
                # Extract neighborhood
                neighborhood = padded_chunk[x-1:x+2, y-1:y+2]
                
                # Compute edge strength using cellular rules
                edge_strength = np.abs(np.sum(neighborhood * kernel))
                
                # Threshold and normalize
                edges_chunk[x-1, y-1] = min(edge_strength, 255)
        
        return edges_chunk
    
    def parallel_edge_detection(self):
        """
        Parallel edge detection using multiprocessing
        
        Returns:
            numpy.ndarray: Detected edges
        """
        # Divide image into chunks
        chunk_size = self.image_size // self.num_processes
        edges = np.zeros_like(self.image, dtype=np.float32)
        
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            # Submit chunks for parallel processing
            futures = []
            for i in range(self.num_processes):
                chunk_start = i * chunk_size
                chunk_end = chunk_start + chunk_size if i < self.num_processes - 1 else self.image_size
                
                future = executor.submit(
                    self.parallel_edge_detection_worker, 
                    chunk_start, 
                    chunk_end
                )
                futures.append(future)
            
            # Collect results
            for i, future in enumerate(futures):
                chunk_start = i * chunk_size
                chunk_end = chunk_start + chunk_size if i < self.num_processes - 1 else self.image_size
                edges[chunk_start:chunk_end] = future.result()
        
        return edges
    
    def visualize_results(self):
        """
        Visualize original and edge-detected images
        """
        # Compute edges
        edges = self.parallel_edge_detection()
        
        # Plot results
        plt.figure(figsize=(12, 5))
        
        plt.subplot(121)
        plt.title('Original Image')
        plt.imshow(self.image, cmap='gray')
        plt.axis('off')
        
        plt.subplot(122)
        plt.title('Parallel Edge Detection')
        plt.imshow(edges, cmap='gray')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# Performance measurement
import time

def measure_performance():
    """
    Measure and compare processing times
    """
    image_sizes = [500, 1000, 2000, 4000]
    process_counts = [1, 2, 4, 8]
    
    performance_data = {}
    
    for size in image_sizes:
        performance_data[size] = {}
        
        for processes in process_counts:
            detector = ParallelEdgeDetector(
                image_size=size, 
                num_processes=processes
            )
            
            start_time = time.time()
            detector.parallel_edge_detection()
            end_time = time.time()
            
            performance_data[size][processes] = end_time - start_time
    
    # Print performance results
    print("Performance Measurements:")
    for size, data in performance_data.items():
        print(f"\nImage Size: {size}x{size}")
        for processes, duration in data.items():
            print(f"  Processes: {processes}, Time: {duration:.4f} seconds")

# Main execution
if __name__ == "__main__":
    # Create and visualize edge detection
    detector = ParallelEdgeDetector(image_size=500)
    detector.visualize_results()
    
    # Measure performance
    measure_performance()
