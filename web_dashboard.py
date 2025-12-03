from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import torch
import numpy as np
from datetime import datetime
import json
import threading
import time

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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'quadramatrix_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', logger=True, engineio_logger=False)

# Global system state
class SystemState:
    def __init__(self):
        self.field_size = 100
        self.oscillator = None
        self.core_field = None
        self.syntropy_engine = None
        self.pattern_module = None
        self.neuroplasticity_manager = None
        self.symbolic_interpreter = None
        
        self.loss_history = []
        self.reward_history = []
        self.variance_history = []
        self.field_mean_history = []
        self.iteration_count = 0
        self.is_running = False
        self.is_initialized = False
        
        self.training_thread = None

state = SystemState()

def emit_update(event, data):
    """Helper to emit updates to all clients"""
    socketio.emit(event, data)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    """Get current system status"""
    return jsonify({
        'initialized': state.is_initialized,
        'running': state.is_running,
        'iteration_count': state.iteration_count,
        'current_metrics': {
            'loss': state.loss_history[-1] if state.loss_history else None,
            'reward': state.reward_history[-1] if state.reward_history else None,
            'variance': state.variance_history[-1] if state.variance_history else None,
            'mean': state.field_mean_history[-1] if state.field_mean_history else None,
            'integrity_strikes': state.neuroplasticity_manager.integrity_strikes if state.neuroplasticity_manager else 0,
            'qtable_size': len(state.oscillator.q_table) if state.oscillator else 0
        },
        'history': {
            'loss': state.loss_history[-50:],
            'reward': state.reward_history[-50:],
            'variance': state.variance_history[-50:],
            'mean': state.field_mean_history[-50:]
        }
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {request.sid}")
    socketio.emit('status_message', {'message': 'Connected to server', 'type': 'info'}, room=request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")

@socketio.on('initialize_system')
def handle_initialize():
    """Initialize the QuadraMatrix system"""
    try:
        socketio.emit('status_message', {'message': 'Initializing QuadraMatrix-Light System...', 'type': 'info'})
        
        state.oscillator = OscillatorySynapseTheory(field_size=state.field_size)
        state.core_field = CoreField(size=state.field_size)
        state.syntropy_engine = SyntropyEngine(num_fields=3, field_size=state.field_size)
        state.pattern_module = PatternModule(n_clusters=3)
        state.neuroplasticity_manager = NeuroplasticityManager(
            state.oscillator, state.core_field, state.syntropy_engine
        )
        symbolic_config = SymbolicConfig()
        state.symbolic_interpreter = SymbolicPredictiveInterpreter(
            state.pattern_module, state.core_field, symbolic_config
        )
        
        state.is_initialized = True
        socketio.emit('status_message', {'message': 'System initialized successfully!', 'type': 'success'})
        socketio.emit('system_initialized', {'initialized': True})
        
    except Exception as e:
        socketio.emit('status_message', {'message': f'Initialization failed: {str(e)}', 'type': 'error'})

@socketio.on('start_training')
def handle_start_training():
    """Start the training process"""
    if not state.is_initialized:
        socketio.emit('status_message', {'message': 'System not initialized!', 'type': 'error'})
        return
    
    if state.is_running:
        socketio.emit('status_message', {'message': 'Training already running!', 'type': 'warning'})
        return
    
    state.is_running = True
    socketio.emit('status_message', {'message': 'Starting training...', 'type': 'info'})
    socketio.emit('training_status', {'running': True})
    
    # Start training in a separate thread
    state.training_thread = threading.Thread(target=training_loop, daemon=True)
    state.training_thread.start()

@socketio.on('stop_training')
def handle_stop_training():
    """Stop the training process"""
    state.is_running = False
    socketio.emit('status_message', {'message': 'Stopping training...', 'type': 'info'})
    socketio.emit('training_status', {'running': False})

@socketio.on('reset_system')
def handle_reset():
    """Reset the system"""
    state.is_running = False
    state.loss_history = []
    state.reward_history = []
    state.variance_history = []
    state.field_mean_history = []
    state.iteration_count = 0
    state.oscillator = None
    state.core_field = None
    state.is_initialized = False
    
    socketio.emit('status_message', {'message': 'System reset.', 'type': 'info'})
    socketio.emit('system_reset', {})

def training_loop():
    """Training loop running in separate thread"""
    test_texts = [
        "The quantum field oscillates with harmonic resonance",
        "Neural networks learn patterns through synaptic plasticity",
        "Syntropy represents order emerging from chaos",
        "Consciousness emerges from complex information processing",
        "Machine learning enables adaptive intelligence systems",
        "Spiking neurons communicate through temporal coding",
        "Deep learning architectures model hierarchical representations",
        "Reinforcement learning optimizes sequential decision making",
    ]
    
    try:
        while state.is_running:
            for text in test_texts:
                if not state.is_running:
                    break
                    
                state.iteration_count += 1
                
                # Run training
                feature_vector = state.oscillator.process_streamed_data(text)
                synthetic_data = state.oscillator.generate_synthetic_data(num_samples=10)
                reward = state.oscillator.train(synthetic_data, text, epochs=3)
                
                # Update field
                field_update = feature_vector.cpu().numpy()
                state.core_field.update_with_vibrational_mode(field_update)
                
                # Calculate metrics
                field_state = state.core_field.get_state()
                field_var = torch.var(field_state).item()
                field_mean = torch.mean(field_state).item()
                
                # Get loss from last training
                with torch.no_grad():
                    test_field = state.oscillator.nn1.update_field(synthetic_data[0])
                    loss = (torch.var(test_field) + torch.abs(torch.mean(test_field) - 0.5)).item()
                
                # Store metrics
                state.loss_history.append(loss)
                state.reward_history.append(reward)
                state.variance_history.append(field_var)
                state.field_mean_history.append(field_mean)
                
                # Emit update to clients
                metrics = {
                    'iteration': state.iteration_count,
                    'loss': loss,
                    'reward': reward,
                    'variance': field_var,
                    'mean': field_mean,
                    'integrity_strikes': state.neuroplasticity_manager.integrity_strikes,
                    'qtable_size': len(state.oscillator.q_table),
                    'text': text
                }
                socketio.emit('metrics_update', metrics)
                
                # Regulate syntropy
                state.neuroplasticity_manager.regulate_syntropy()
                
                # Small delay
                time.sleep(0.5)
                
    except Exception as e:
        socketio.emit('status_message', {'message': f'Training error: {str(e)}', 'type': 'error'})
        state.is_running = False
        socketio.emit('training_status', {'running': False})

if __name__ == '__main__':
    print("Starting QuadraMatrix Benchmark Dashboard...")
    print("Open your browser to: http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
