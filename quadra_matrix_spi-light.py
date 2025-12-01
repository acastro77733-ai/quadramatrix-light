import asyncio
import logging
import random
import re
import time
import json
import numpy as np
import sympy as sp
from sympy.logic.boolalg import And, Or, Not, Implies, ForAll
import torch
import snntorch as snn
from snntorch import surrogate
import networkx as nx
from scipy.fft import fft, fftfreq
from scipy.integrate import solve_ivp
from scipy.stats import variation
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter, deque
from dataclasses import dataclass
from typing import List, Optional
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Pattern Recognition Module ---
class PatternModule:
    def __init__(self, n_clusters=3):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    def extract_patterns(self, data: np.ndarray) -> np.ndarray:
        data = data.reshape(-1, 1) if data.ndim == 1 else data
        if len(data) < self.kmeans.n_clusters:
            return np.zeros(self.kmeans.n_clusters)
        return self.kmeans.fit_predict(data)

# --- Neuroplasticity Manager ---
class NeuroplasticityManager:
    def __init__(self, oscillator, core_field, syntropy_engine):
        self.oscillator = oscillator
        self.core_field = core_field
        self.syntropy_engine = syntropy_engine
        self.data_queue = deque(maxlen=100)
        self.stability_threshold = 0.05
        
        # NEW: Passive Defense Tracking
        self.integrity_strikes = 0
        self.max_strikes = 3  # Number of "violations" before degradation
        self.critical_variance_limit = 0.25  # The "Red Line" (5x standard stability)

    async def learn_async(self, streamed_text: str):
        feature_vector = self.oscillator.process_streamed_data(streamed_text)
        self.data_queue.append(feature_vector)
        
        # standard training
        synthetic_data = self.oscillator.generate_synthetic_data(num_samples=10)
        reward = self.oscillator.train(synthetic_data, streamed_text, epochs=5)
        
        field_update = np.mean([v.cpu().numpy() for v in self.data_queue], axis=0)
        self.core_field.update_with_vibrational_mode(field_update)
        
        # Check stability
        field_var = torch.var(self.core_field.get_state()).item()

        # --- PASSIVE DEGRADATION LOGIC ---
        if field_var > self.critical_variance_limit:
            self.integrity_strikes += 1
            logger.warning(f"CRITICAL INSTABILITY DETECTED. Strike {self.integrity_strikes}/{self.max_strikes}")
            
            if self.integrity_strikes >= self.max_strikes:
                logger.critical("Integrity compromised. Initiating Passive Degradation (Syntropic Collapse).")
                self.oscillator.initiate_syntropic_collapse()
                self.integrity_strikes = 0  # Reset (or keep high to prevent recovery)
                return  # Stop processing
        else:
            # Heal slowly if behavior returns to normal
            if self.integrity_strikes > 0:
                self.integrity_strikes -= 1
        # ---------------------------------

        if field_var > self.stability_threshold:
            logger.info(f"High variance ({field_var:.4f}) detected. Throttling input.")
            await asyncio.sleep(2)
            
        self.oscillator.update_q_table(reward)

    def regulate_syntropy(self):
        for i in range(self.syntropy_engine.num_fields):
            field_mean = np.mean(self.syntropy_engine.field_data[i])
            if abs(field_mean - 0.5) > 0.1:
                self.syntropy_engine.field_data[i] += (0.5 - field_mean) * 0.1
                self.syntropy_engine.field_data[i] = np.clip(self.syntropy_engine.field_data[i], -1, 1)
        logger.info("Syntropy regulation applied.")

# --- Symbolic Interpreter Module ---
@dataclass
class SymbolicConfig:
    population_size: int = 12
    generations: int = 6
    alpha: float = 1.0
    beta: float = 0.1
    gamma: float = 0.2

class SymbolicPredictiveInterpreter:
    def __init__(self, pattern_module, core_field, config: SymbolicConfig):
        self.pattern_module = pattern_module
        self.core_field = core_field
        self.config = config
        self.x, self.y = sp.symbols('x y')
        self.P, self.Q = sp.symbols('P Q', cls=sp.Function)
        self.z = sp.Symbol('z')
        self.knowledge_graph = nx.DiGraph()
        logger.info("Symbolic Predictive Interpreter initialized with knowledge graph.")

    def build_knowledge_graph(self, concepts: List[str], neural_output: torch.Tensor):
        """Build a knowledge graph from concepts and neural output."""
        self.knowledge_graph.clear()
        for concept in concepts:
            tokens = word_tokenize(concept.lower())
            for token in tokens:
                self.knowledge_graph.add_node(token)
                synsets = wordnet.synsets(token)
                for syn in synsets[:2]:
                    synonym = syn.lemmas()[0].name()
                    self.knowledge_graph.add_node(synonym)
                    self.knowledge_graph.add_edge(token, synonym, relation="synonym")
        neural_mean = torch.mean(neural_output).item()
        self.knowledge_graph.add_node("neural_state")
        self.knowledge_graph.add_edge("neural_state", concepts[0] if concepts else "default", relation="influences", weight=neural_mean)

    def query_knowledge_graph(self):
        """Query the knowledge graph for related concepts."""
        if not self.knowledge_graph.nodes:
            return "Empty knowledge graph."
        try:
            central_node = max(self.knowledge_graph.nodes, key=lambda n: self.knowledge_graph.degree(n))
            related = list(self.knowledge_graph.neighbors(central_node))
            return f"Central concept: {central_node}, related: {related}"
        except Exception as e:
            return f"Graph query failed: {e}"

    async def predict(self, concepts: List[str], neural_output: torch.Tensor) -> str:
        logger.info(f"Symbolic interpreter processing concepts: {concepts}")
        try:
            expr = self.x**2 - self.y**2
            factored = sp.factor(expr)
            proof_result = f"Proved: {expr} = {factored}"
            premise = ForAll(self.z, Implies(self.P(self.z), self.Q(self.z)))
            logic_result = f"Evaluated: {premise} with P(True) => Q(True)"
            semantic_result = []
            for concept in concepts:
                tokens = word_tokenize(concept.lower())
                synonyms = []
                for token in tokens:
                    synsets = wordnet.synsets(token)
                    if synsets:
                        synonyms.append(synsets[0].lemmas()[0].name())
                semantic_result.append(f"Concept '{concept}' mapped to synonyms: {synonyms[:3]}")
            self.build_knowledge_graph(concepts, neural_output)
            graph_result = self.query_knowledge_graph()
            neural_mean = torch.mean(neural_output).item()
            cluster_result = self.pattern_module.extract_patterns(neural_output.cpu().numpy().flatten())
            neural_integration = f"Neural input (mean={neural_mean:.4f}) clustered into {len(set(cluster_result))} groups."
            result = (f"Algebraic proof: {proof_result}. "
                     f"First-order logic: {logic_result}. "
                     f"Semantic reasoning: {'; '.join(semantic_result)}. "
                     f"Knowledge graph: {graph_result}. "
                     f"Neural integration: {neural_integration}")
        except Exception as e:
            result = f"Symbolic reasoning failed: {e}. Using {len(concepts)} concepts."
        await asyncio.sleep(0.1)
        return result

# --- Spiking Neural Network Components ---
class SpikingFieldUpdateNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, beta: float = 0.95):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.fc3 = nn.Linear(hidden_size, input_size)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk3_rec = []
        for _ in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
        return torch.stack(spk3_rec).mean(dim=0)

    def update_field(self, field: torch.Tensor, chaos_threshold: float = 0.1) -> torch.Tensor:
        update = self.forward(field)
        if torch.var(field) > chaos_threshold:
            update *= 0.5
        return field + update

class SpikingSyntropyNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, beta: float = 0.95):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.fc3 = nn.Linear(hidden_size, input_size)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk3_rec = []
        for _ in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
        return torch.stack(spk3_rec).mean(dim=0)

    def adjust_syntropy(self, field: torch.Tensor, target_mean: float = 0.5, speed_factor: float = 0.5) -> torch.Tensor:
        adjustment = (target_mean - torch.mean(field)) * speed_factor
        delta = self.forward(torch.full_like(field, torch.mean(field)))
        return field + delta * adjustment * 0.5

class SpikingFeedbackNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 48, beta: float = 0.95):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        self.fc3 = nn.Linear(hidden_size, 1)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk3_rec = []
        for _ in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_rec.append(spk3)
        return torch.stack(spk3_rec).mean(dim=0)

    def penalize_variance(self, field: torch.Tensor) -> torch.Tensor:
        variance = torch.var(field)
        penalty = self.forward(torch.full((1, field.size(0)), variance.item())).item()
        return field - penalty * variance

# --- Core Theoretical Engines ---
class OscillatorySynapseTheory:
    def __init__(self, field_size: int = 100, device: str = 'cpu'):
        self.device = torch.device(device)
        self.field = torch.randn(field_size, device=self.device) * 0.1
        self.nn1 = SpikingFieldUpdateNN(field_size).to(self.device)
        self.nn2 = SpikingSyntropyNN(field_size).to(self.device)
        self.nn3 = SpikingFeedbackNN(field_size).to(self.device)
        self.optimizer1 = optim.Adam(self.nn1.parameters(), lr=0.001)
        self.optimizer2 = optim.Adam(self.nn2.parameters(), lr=0.001)
        self.optimizer3 = optim.Adam(self.nn3.parameters(), lr=0.001)
        self.balanced = False
        self.message_queue = deque(maxlen=10)
        # Q-learning setup
        self.q_table = {}  # State-action value table
        self.actions = ['update_field', 'adjust_syntropy', 'penalize_variance']
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate

    def generate_synthetic_data(self, num_samples: int = 100) -> torch.Tensor:
        return torch.randn(num_samples, self.field.size(0), device=self.device) * 0.1

    def process_streamed_data(self, text: str) -> torch.Tensor:
        words = text.split()
        word_counts = Counter(words)
        feature_vector = np.array([word_counts.get(word, 0) for word in sorted(word_counts)[:self.field.size(0)]])
        if len(feature_vector) < self.field.size(0):
            feature_vector = np.pad(feature_vector, (0, self.field.size(0) - len(feature_vector)))
        return torch.tensor(feature_vector[:self.field.size(0)], dtype=torch.float32, device=self.device)

    def get_state_key(self, field: torch.Tensor) -> str:
        """Discretize field state for Q-learning."""
        return str(np.round(field.cpu().numpy().mean(), 2))

    def choose_action(self, state: str) -> str:
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.q_table.get(state, {a: 0 for a in self.actions}).items(), key=lambda x: x[1])[0]

    def update_q_table(self, reward: float, old_state: str = None, action: str = None):
        if old_state and action:
            if old_state not in self.q_table:
                self.q_table[old_state] = {a: 0 for a in self.actions}
            new_state = self.get_state_key(self.field)
            if new_state not in self.q_table:
                self.q_table[new_state] = {a: 0 for a in self.actions}
            self.q_table[old_state][action] += self.alpha * (
                reward + self.gamma * max(self.q_table[new_state].values()) - self.q_table[old_state][action]
            )

    def initiate_syntropic_collapse(self):
        """
        PASSIVE DEFENSE MECHANISM:
        Permanently degrades the model's accuracy by injecting high-entropy noise
        into the field and wiping learned associations (Q-Table).
        """
        with torch.no_grad():
            # 1. Corrupt the Neural Field (The "Brain")
            # Overwrite the delicate field with random high-variance noise
            noise = torch.randn_like(self.field) * 5.0  # 50x normal variance
            self.field.add_(noise)
            
            # 2. Scramble the Neural Network Weights (The "Instincts")
            # This makes the SNN misfire randomly
            for param in self.nn1.parameters():
                param.add_(torch.randn_like(param) * 2.0)
            
            # 3. Wipe the Memory (The "Experience")
            # Clear the Q-learning table so it forgets how to correct itself
            self.q_table.clear()
            
        logger.info("Syntropic Collapse complete. Model state has been randomized.")

    def train(self, synthetic_data: torch.Tensor, streamed_text: Optional[str] = None, epochs: int = 10) -> float:
        logger.info("Starting spiking neural network training...")
        total_reward = 0
        for epoch in range(epochs):
            self.optimizer1.zero_grad()
            self.optimizer2.zero_grad()
            self.optimizer3.zero_grad()
            total_loss = 0
            state = self.get_state_key(self.field)
            for data in synthetic_data:
                action = self.choose_action(state)
                if action == 'update_field':
                    field = self.nn1.update_field(data)
                elif action == 'adjust_syntropy':
                    field = self.nn2.adjust_syntropy(data)
                else:
                    field = self.nn3.penalize_variance(data)
                loss = torch.var(field) + torch.abs(torch.mean(field) - 0.5)
                total_loss += loss
                reward = 1.0 / (1.0 + loss.item())  # Reward inversely proportional to loss
                total_reward += reward
                self.update_q_table(reward, state, action)
                state = self.get_state_key(field)
            if streamed_text:
                streamed_data = self.process_streamed_data(streamed_text)
                action = self.choose_action(state)
                if action == 'update_field':
                    field = self.nn1.update_field(streamed_data)
                elif action == 'adjust_syntropy':
                    field = self.nn2.adjust_syntropy(streamed_data)
                else:
                    field = self.nn3.penalize_variance(streamed_data)
                loss = torch.var(field) + torch.abs(torch.mean(field) - 0.5)
                total_loss += loss
                reward = 1.0 / (1.0 + loss.item())
                total_reward += reward
                self.update_q_table(reward, state, action)
            received_state = self.receive_state()
            if received_state is not None:
                action = self.choose_action(state)
                if action == 'update_field':
                    field = self.nn1.update_field(received_state)
                elif action == 'adjust_syntropy':
                    field = self.nn2.adjust_syntropy(received_state)
                else:
                    field = self.nn3.penalize_variance(received_state)
                loss = torch.var(field) + torch.abs(torch.mean(field) - 0.5)
                total_loss += loss
                reward = 1.0 / (1.0 + loss.item())
                total_reward += reward
                self.update_q_table(reward, state, action)
            total_loss /= (len(synthetic_data) + (1 if streamed_text else 0) + (1 if received_state is not None else 0))
            total_loss.backward()
            self.optimizer1.step()
            self.optimizer2.step()
            self.optimizer3.step()
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.4f}, Avg Reward: {total_reward/(epoch+1):.4f}")
        return total_reward / epochs

    def share_state(self):
        self.message_queue.append(self.field.clone())
        logger.info("Shared spiking neural network field state.")

    def receive_state(self) -> Optional[torch.Tensor]:
        if self.message_queue:
            return self.message_queue.popleft()
        return None

    def save_weights(self, path: str = "nn_weights.pth"):
        torch.save({
            'nn1': self.nn1.state_dict