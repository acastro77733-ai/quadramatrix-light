from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
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

PERSIST_DIR = Path('storage')
STATE_FILE = PERSIST_DIR / 'dashboard_state.json'
WEIGHTS_FILE = PERSIST_DIR / 'dashboard_weights.pth'
LOG_LIMIT = 300
WEIGHT_SAVE_INTERVAL = 5
DEFAULT_TRAINING_TEXT = "No training in progress..."

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
        self.status_log = []
        self.current_training_text = DEFAULT_TRAINING_TEXT
        self.persistence_lock = threading.Lock()
        self.pending_snapshot = None
        self.last_weight_save_iteration = 0

state = SystemState()

def ensure_persist_dir():
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)

def tensor_to_list(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return value

def save_persistent_state():
    with state.persistence_lock:
        ensure_persist_dir()
        snapshot = {
            'timestamp': datetime.utcnow().isoformat(),
            'is_initialized': state.is_initialized,
            'is_running': state.is_running,
            'iteration_count': state.iteration_count,
            'loss_history': state.loss_history,
            'reward_history': state.reward_history,
            'variance_history': state.variance_history,
            'mean_history': state.field_mean_history,
            'status_log': state.status_log[-LOG_LIMIT:],
            'current_text': state.current_training_text,
            'integrity_strikes': state.neuroplasticity_manager.integrity_strikes if state.neuroplasticity_manager else 0,
        }
        if state.oscillator:
            snapshot['oscillator'] = {
                'field': tensor_to_list(state.oscillator.field),
                'q_table': state.oscillator.q_table,
            }
        if state.core_field:
            snapshot['core_field_state'] = tensor_to_list(state.core_field.get_state())
        if state.syntropy_engine:
            snapshot['syntropy_fields'] = [tensor_to_list(field) for field in state.syntropy_engine.field_data]
        with STATE_FILE.open('w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=2)

def load_persistent_state():
    if not STATE_FILE.exists():
        return None
    with STATE_FILE.open('r', encoding='utf-8') as f:
        return json.load(f)

def apply_snapshot_to_state(snapshot):
    state.loss_history = snapshot.get('loss_history', [])
    state.reward_history = snapshot.get('reward_history', [])
    state.variance_history = snapshot.get('variance_history', [])
    state.field_mean_history = snapshot.get('mean_history', [])
    state.iteration_count = snapshot.get('iteration_count', 0)
    state.status_log = snapshot.get('status_log', [])[-LOG_LIMIT:]
    state.current_training_text = snapshot.get('current_text', DEFAULT_TRAINING_TEXT)
    state.is_running = False
    state.pending_snapshot = snapshot

def record_status(message, level='info', emit=True, persist=True):
    entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'message': message,
        'type': level
    }
    state.status_log.append(entry)
    state.status_log = state.status_log[-LOG_LIMIT:]
    if emit:
        socketio.emit('status_message', {'message': message, 'type': level, 'timestamp': entry['timestamp']})
    if persist:
        save_persistent_state()

def persist_weights(force=False):
    if not state.oscillator:
        return
    if not force and state.iteration_count - state.last_weight_save_iteration < WEIGHT_SAVE_INTERVAL:
        return
    try:
        ensure_persist_dir()
        state.oscillator.save_weights(str(WEIGHTS_FILE))
        state.last_weight_save_iteration = state.iteration_count
    except Exception as exc:
        record_status(f'Failed to save weights: {exc}', 'error', emit=False, persist=False)

def restore_components_from_snapshot(snapshot):
    if not snapshot:
        return
    if state.core_field and snapshot.get('core_field_state') is not None:
        restored_core = torch.tensor(snapshot['core_field_state'], dtype=torch.float32, device=state.core_field.device)
        state.core_field.state = restored_core
    oscillator_state = snapshot.get('oscillator')
    if state.oscillator and oscillator_state:
        if oscillator_state.get('field') is not None:
            state.oscillator.field = torch.tensor(oscillator_state['field'], dtype=torch.float32, device=state.oscillator.device)
        if oscillator_state.get('q_table') is not None:
            state.oscillator.q_table = oscillator_state['q_table']
    if state.syntropy_engine and snapshot.get('syntropy_fields'):
        state.syntropy_engine.field_data = [np.array(field) for field in snapshot['syntropy_fields']]
    if state.neuroplasticity_manager:
        state.neuroplasticity_manager.integrity_strikes = snapshot.get('integrity_strikes', 0)
    if state.oscillator and WEIGHTS_FILE.exists():
        try:
            state.oscillator.load_weights(str(WEIGHTS_FILE))
        except Exception as exc:
            record_status(f'Failed to load persisted weights: {exc}', 'error', emit=False, persist=False)

def initialize_components(snapshot=None):
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
    if snapshot:
        restore_components_from_snapshot(snapshot)
    state.pending_snapshot = None
    state.is_initialized = True
    save_persistent_state()

def build_client_snapshot():
    return {
        'initialized': state.is_initialized,
        'running': state.is_running,
        'iteration_count': state.iteration_count,
        'history': {
            'loss': state.loss_history,
            'reward': state.reward_history,
            'variance': state.variance_history,
            'mean': state.field_mean_history
        },
        'status_log': state.status_log[-LOG_LIMIT:],
        'current_text': state.current_training_text,
        'current_metrics': {
            'loss': state.loss_history[-1] if state.loss_history else None,
            'reward': state.reward_history[-1] if state.reward_history else None,
            'variance': state.variance_history[-1] if state.variance_history else None,
            'mean': state.field_mean_history[-1] if state.field_mean_history else None,
            'integrity_strikes': state.neuroplasticity_manager.integrity_strikes if state.neuroplasticity_manager else 0,
            'qtable_size': len(state.oscillator.q_table) if state.oscillator else 0
        }
    }

def bootstrap_from_disk():
    snapshot = load_persistent_state()
    if not snapshot:
        return
    apply_snapshot_to_state(snapshot)
    if snapshot.get('is_initialized'):
        try:
            initialize_components(snapshot)
            record_status('Restored QuadraMatrix state from disk.', 'success', emit=False)
        except Exception as exc:
            record_status(f'Failed to restore persisted state: {exc}', 'error', emit=False)
    else:
        save_persistent_state()

bootstrap_from_disk()

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
        },
        'status_log': state.status_log[-LOG_LIMIT:],
        'current_text': state.current_training_text
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {request.sid}")
    socketio.emit('status_message', {
        'message': 'Connected to server',
        'type': 'info',
        'timestamp': datetime.utcnow().isoformat()
    }, room=request.sid)
    socketio.emit('initial_state', build_client_snapshot(), room=request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {request.sid}")

@socketio.on('initialize_system')
def handle_initialize():
    """Initialize the QuadraMatrix system"""
    if state.is_initialized:
        record_status('System already initialized. Ready for training.', 'warning')
        socketio.emit('system_initialized', {'initialized': True})
        return
    try:
        record_status('Initializing QuadraMatrix-Light System...', 'info')
        initialize_components(state.pending_snapshot)
        record_status('System initialized successfully!', 'success')
        socketio.emit('system_initialized', {'initialized': True})
    except Exception as e:
        record_status(f'Initialization failed: {str(e)}', 'error')

@socketio.on('start_training')
def handle_start_training():
    """Start the training process"""
    if not state.is_initialized:
        record_status('System not initialized!', 'error')
        return
    
    if state.is_running:
        record_status('Training already running!', 'warning')
        return
    
    state.is_running = True
    record_status('Starting training...', 'info')
    socketio.emit('training_status', {'running': True})
    save_persistent_state()
    
    # Start training in a separate thread
    state.training_thread = threading.Thread(target=training_loop, daemon=True)
    state.training_thread.start()

@socketio.on('stop_training')
def handle_stop_training():
    """Stop the training process"""
    state.is_running = False
    state.current_training_text = DEFAULT_TRAINING_TEXT
    record_status('Stopping training...', 'info')
    socketio.emit('training_status', {'running': False})
    save_persistent_state()

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
    state.syntropy_engine = None
    state.pattern_module = None
    state.neuroplasticity_manager = None
    state.symbolic_interpreter = None
    state.is_initialized = False
    state.current_training_text = DEFAULT_TRAINING_TEXT
    state.pending_snapshot = None
    if WEIGHTS_FILE.exists():
        WEIGHTS_FILE.unlink()
    save_persistent_state()
    
    record_status('System reset.', 'info')
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
                state.current_training_text = text
                
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
                persist_weights()
                save_persistent_state()
                
                # Regulate syntropy
                state.neuroplasticity_manager.regulate_syntropy()
                
                # Small delay
                time.sleep(0.5)
        save_persistent_state()
                
    except Exception as e:
        record_status(f'Training error: {str(e)}', 'error')
        state.is_running = False
        socketio.emit('training_status', {'running': False})
        save_persistent_state()

if __name__ == '__main__':
    print("Starting QuadraMatrix Benchmark Dashboard...")
    print("Open your browser to: http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
