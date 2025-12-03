import tkinter as tk
from tkinter import ttk, scrolledtext
import asyncio
import threading
import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import logging

# Import the main components
from quadra_matrix_spi_light import (
    OscillatorySynapseTheory,
    CoreField,
    SyntropyEngine,
    PatternModule,
    NeuroplasticityManager,
    SymbolicConfig,
    SymbolicPredictiveInterpreter
)

class QuadraMatrixGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("QuadraMatrix-Light Benchmark Dashboard")
        self.root.geometry("1400x900")
        
        # Initialize system components
        self.field_size = 100
        self.oscillator = None
        self.core_field = None
        self.syntropy_engine = None
        self.pattern_module = None
        self.neuroplasticity_manager = None
        self.symbolic_interpreter = None
        
        # Benchmark data storage
        self.loss_history = []
        self.reward_history = []
        self.variance_history = []
        self.field_mean_history = []
        self.iteration_count = 0
        self.is_running = False
        
        # Setup logging to capture in GUI
        self.setup_logging()
        
        # Create GUI components
        self.create_widgets()
        
    def setup_logging(self):
        """Setup logging to capture in GUI"""
        self.log_handler = logging.Handler()
        self.log_handler.setLevel(logging.INFO)
        self.log_handler.emit = self.log_message
        logging.getLogger().addHandler(self.log_handler)
        
    def log_message(self, record):
        """Capture log messages"""
        msg = self.log_handler.format(record)
        if hasattr(self, 'log_text'):
            self.log_text.insert(tk.END, msg + '\n')
            self.log_text.see(tk.END)
        
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main container with two columns
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Left panel - Controls and metrics
        left_panel = ttk.Frame(main_frame, padding="5")
        left_panel.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Right panel - Graphs
        right_panel = ttk.Frame(main_frame, padding="5")
        right_panel.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Control panel
        control_frame = ttk.LabelFrame(left_panel, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.start_btn = ttk.Button(control_frame, text="Initialize System", command=self.initialize_system)
        self.start_btn.grid(row=0, column=0, padx=5, pady=5)
        
        self.train_btn = ttk.Button(control_frame, text="Start Training", command=self.start_training, state=tk.DISABLED)
        self.train_btn.grid(row=0, column=1, padx=5, pady=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=2, padx=5, pady=5)
        
        self.reset_btn = ttk.Button(control_frame, text="Reset", command=self.reset_system)
        self.reset_btn.grid(row=0, column=3, padx=5, pady=5)
        
        # Metrics panel
        metrics_frame = ttk.LabelFrame(left_panel, text="Real-time Metrics", padding="10")
        metrics_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N), pady=5)
        
        metrics_labels = [
            ("Iterations:", "iteration_label"),
            ("Current Loss:", "loss_label"),
            ("Average Reward:", "reward_label"),
            ("Field Variance:", "variance_label"),
            ("Field Mean:", "mean_label"),
            ("Integrity Strikes:", "strikes_label"),
            ("Q-Table Size:", "qtable_label"),
        ]
        
        for idx, (label_text, attr_name) in enumerate(metrics_labels):
            ttk.Label(metrics_frame, text=label_text, font=("Arial", 10, "bold")).grid(row=idx, column=0, sticky=tk.W, pady=3)
            label = ttk.Label(metrics_frame, text="--", font=("Arial", 10))
            label.grid(row=idx, column=1, sticky=tk.W, padx=10, pady=3)
            setattr(self, attr_name, label)
        
        # Status panel
        status_frame = ttk.LabelFrame(left_panel, text="System Status", padding="10")
        status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N), pady=5)
        
        self.status_text = scrolledtext.ScrolledText(status_frame, width=50, height=8, wrap=tk.WORD)
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Log panel
        log_frame = ttk.LabelFrame(left_panel, text="Training Log", padding="10")
        log_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        left_panel.rowconfigure(3, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, width=50, height=15, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        
        # Graph panel
        self.create_graphs(right_panel)
        
    def create_graphs(self, parent):
        """Create matplotlib graphs"""
        # Create figure with subplots
        self.fig = Figure(figsize=(10, 8), dpi=100)
        
        # Loss plot
        self.ax1 = self.fig.add_subplot(2, 2, 1)
        self.ax1.set_title("Training Loss Over Time")
        self.ax1.set_xlabel("Iteration")
        self.ax1.set_ylabel("Loss")
        self.ax1.grid(True, alpha=0.3)
        
        # Reward plot
        self.ax2 = self.fig.add_subplot(2, 2, 2)
        self.ax2.set_title("Average Reward Over Time")
        self.ax2.set_xlabel("Iteration")
        self.ax2.set_ylabel("Reward")
        self.ax2.grid(True, alpha=0.3)
        
        # Variance plot
        self.ax3 = self.fig.add_subplot(2, 2, 3)
        self.ax3.set_title("Field Variance Over Time")
        self.ax3.set_xlabel("Iteration")
        self.ax3.set_ylabel("Variance")
        self.ax3.grid(True, alpha=0.3)
        
        # Field mean plot
        self.ax4 = self.fig.add_subplot(2, 2, 4)
        self.ax4.set_title("Field Mean Over Time")
        self.ax4.set_xlabel("Iteration")
        self.ax4.set_ylabel("Mean Value")
        self.ax4.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        
    def initialize_system(self):
        """Initialize the QuadraMatrix system"""
        self.update_status("Initializing QuadraMatrix-Light System...")
        
        try:
            self.oscillator = OscillatorySynapseTheory(field_size=self.field_size)
            self.core_field = CoreField(size=self.field_size)
            self.syntropy_engine = SyntropyEngine(num_fields=3, field_size=self.field_size)
            self.pattern_module = PatternModule(n_clusters=3)
            self.neuroplasticity_manager = NeuroplasticityManager(
                self.oscillator, self.core_field, self.syntropy_engine
            )
            symbolic_config = SymbolicConfig()
            self.symbolic_interpreter = SymbolicPredictiveInterpreter(
                self.pattern_module, self.core_field, symbolic_config
            )
            
            self.update_status("System initialized successfully!")
            self.start_btn.config(state=tk.DISABLED)
            self.train_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            self.update_status(f"Initialization failed: {str(e)}")
            
    def start_training(self):
        """Start the training process"""
        self.is_running = True
        self.train_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.update_status("Starting training...")
        
        # Run training in a separate thread
        thread = threading.Thread(target=self.run_training_loop, daemon=True)
        thread.start()
        
    def run_training_loop(self):
        """Training loop running in separate thread"""
        test_texts = [
            "The quantum field oscillates with harmonic resonance",
            "Neural networks learn patterns through synaptic plasticity",
            "Syntropy represents order emerging from chaos",
            "Consciousness emerges from complex information processing",
            "Machine learning enables adaptive intelligence systems",
            "Spiking neurons communicate through temporal coding",
        ]
        
        try:
            while self.is_running:
                for text in test_texts:
                    if not self.is_running:
                        break
                        
                    self.iteration_count += 1
                    
                    # Run training
                    feature_vector = self.oscillator.process_streamed_data(text)
                    synthetic_data = self.oscillator.generate_synthetic_data(num_samples=10)
                    reward = self.oscillator.train(synthetic_data, text, epochs=3)
                    
                    # Update field
                    field_update = feature_vector.cpu().numpy()
                    self.core_field.update_with_vibrational_mode(field_update)
                    
                    # Calculate metrics
                    field_state = self.core_field.get_state()
                    field_var = torch.var(field_state).item()
                    field_mean = torch.mean(field_state).item()
                    
                    # Get loss from last training
                    with torch.no_grad():
                        test_field = self.oscillator.nn1.update_field(synthetic_data[0])
                        loss = (torch.var(test_field) + torch.abs(torch.mean(test_field) - 0.5)).item()
                    
                    # Store metrics
                    self.loss_history.append(loss)
                    self.reward_history.append(reward)
                    self.variance_history.append(field_var)
                    self.field_mean_history.append(field_mean)
                    
                    # Update GUI
                    self.root.after(0, self.update_metrics)
                    self.root.after(0, self.update_graphs)
                    
                    # Regulate syntropy
                    self.neuroplasticity_manager.regulate_syntropy()
                    
                    # Small delay
                    import time
                    time.sleep(0.5)
                    
        except Exception as e:
            self.root.after(0, self.update_status, f"Training error: {str(e)}")
            self.root.after(0, self.stop_training)
            
    def stop_training(self):
        """Stop the training process"""
        self.is_running = False
        self.train_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.update_status("Training stopped.")
        
    def reset_system(self):
        """Reset the system"""
        self.is_running = False
        self.loss_history = []
        self.reward_history = []
        self.variance_history = []
        self.field_mean_history = []
        self.iteration_count = 0
        
        self.oscillator = None
        self.core_field = None
        
        self.start_btn.config(state=tk.NORMAL)
        self.train_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        
        # Clear graphs
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        self.ax1.set_title("Training Loss Over Time")
        self.ax1.set_xlabel("Iteration")
        self.ax1.set_ylabel("Loss")
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.set_title("Average Reward Over Time")
        self.ax2.set_xlabel("Iteration")
        self.ax2.set_ylabel("Reward")
        self.ax2.grid(True, alpha=0.3)
        
        self.ax3.set_title("Field Variance Over Time")
        self.ax3.set_xlabel("Iteration")
        self.ax3.set_ylabel("Variance")
        self.ax3.grid(True, alpha=0.3)
        
        self.ax4.set_title("Field Mean Over Time")
        self.ax4.set_xlabel("Iteration")
        self.ax4.set_ylabel("Mean Value")
        self.ax4.grid(True, alpha=0.3)
        
        self.canvas.draw()
        
        # Clear metrics
        self.iteration_label.config(text="--")
        self.loss_label.config(text="--")
        self.reward_label.config(text="--")
        self.variance_label.config(text="--")
        self.mean_label.config(text="--")
        self.strikes_label.config(text="--")
        self.qtable_label.config(text="--")
        
        self.update_status("System reset.")
        
    def update_metrics(self):
        """Update the metrics display"""
        if not self.loss_history:
            return
            
        self.iteration_label.config(text=str(self.iteration_count))
        self.loss_label.config(text=f"{self.loss_history[-1]:.4f}")
        self.reward_label.config(text=f"{self.reward_history[-1]:.4f}")
        self.variance_label.config(text=f"{self.variance_history[-1]:.6f}")
        self.mean_label.config(text=f"{self.field_mean_history[-1]:.4f}")
        
        if self.neuroplasticity_manager:
            self.strikes_label.config(text=str(self.neuroplasticity_manager.integrity_strikes))
        
        if self.oscillator:
            self.qtable_label.config(text=str(len(self.oscillator.q_table)))
        
    def update_graphs(self):
        """Update the matplotlib graphs"""
        if not self.loss_history:
            return
            
        iterations = list(range(1, len(self.loss_history) + 1))
        
        # Update loss plot
        self.ax1.clear()
        self.ax1.plot(iterations, self.loss_history, 'b-', linewidth=2)
        self.ax1.set_title("Training Loss Over Time")
        self.ax1.set_xlabel("Iteration")
        self.ax1.set_ylabel("Loss")
        self.ax1.grid(True, alpha=0.3)
        
        # Update reward plot
        self.ax2.clear()
        self.ax2.plot(iterations, self.reward_history, 'g-', linewidth=2)
        self.ax2.set_title("Average Reward Over Time")
        self.ax2.set_xlabel("Iteration")
        self.ax2.set_ylabel("Reward")
        self.ax2.grid(True, alpha=0.3)
        
        # Update variance plot
        self.ax3.clear()
        self.ax3.plot(iterations, self.variance_history, 'r-', linewidth=2)
        self.ax3.axhline(y=0.05, color='orange', linestyle='--', label='Stability Threshold')
        self.ax3.axhline(y=0.25, color='red', linestyle='--', label='Critical Limit')
        self.ax3.set_title("Field Variance Over Time")
        self.ax3.set_xlabel("Iteration")
        self.ax3.set_ylabel("Variance")
        self.ax3.legend()
        self.ax3.grid(True, alpha=0.3)
        
        # Update mean plot
        self.ax4.clear()
        self.ax4.plot(iterations, self.field_mean_history, 'm-', linewidth=2)
        self.ax4.axhline(y=0.5, color='gray', linestyle='--', label='Target')
        self.ax4.set_title("Field Mean Over Time")
        self.ax4.set_xlabel("Iteration")
        self.ax4.set_ylabel("Mean Value")
        self.ax4.legend()
        self.ax4.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
        
    def update_status(self, message):
        """Update status text"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        
def main():
    root = tk.Tk()
    app = QuadraMatrixGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
