"""
Synthetic Spacecraft Telemetry Data Generator
=============================================

Generates realistic spacecraft telemetry data with injected anomalies
for testing and demonstration purposes.

Telemetry channels simulated:
- Temperature sensors
- Power system metrics  
- Attitude control data
- Communication signal strength
- Battery metrics
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


@dataclass
class AnomalyType:
    """Defines different types of spacecraft anomalies."""
    name: str
    description: str
    
    
ANOMALY_TYPES = {
    "spike": AnomalyType("spike", "Sudden sharp increase/decrease"),
    "drift": AnomalyType("drift", "Gradual deviation from normal"),
    "step": AnomalyType("step", "Sudden level shift"),
    "oscillation": AnomalyType("oscillation", "Unusual periodic behavior"),
    "dropout": AnomalyType("dropout", "Signal loss or zero values"),
    "noise": AnomalyType("noise", "Increased noise/variance"),
}


class SpacecraftTelemetryGenerator:
    """
    Generates synthetic spacecraft telemetry data with realistic patterns
    and injected anomalies.
    """
    
    # Telemetry channel definitions with typical ranges
    CHANNELS = {
        "temp_battery": {"min": 15, "max": 45, "unit": "°C", "noise": 0.5},
        "temp_payload": {"min": -10, "max": 35, "unit": "°C", "noise": 0.8},
        "temp_solar_panel": {"min": -50, "max": 100, "unit": "°C", "noise": 2.0},
        "voltage_bus": {"min": 26, "max": 32, "unit": "V", "noise": 0.2},
        "current_draw": {"min": 0.5, "max": 5.0, "unit": "A", "noise": 0.1},
        "battery_soc": {"min": 20, "max": 100, "unit": "%", "noise": 0.5},
        "attitude_roll": {"min": -5, "max": 5, "unit": "deg", "noise": 0.3},
        "attitude_pitch": {"min": -5, "max": 5, "unit": "deg", "noise": 0.3},
        "attitude_yaw": {"min": -5, "max": 5, "unit": "deg", "noise": 0.3},
        "signal_strength": {"min": -100, "max": -60, "unit": "dBm", "noise": 1.0},
    }
    
    def __init__(
        self,
        n_timesteps: int = 5000,
        sampling_rate: float = 1.0,  # Hz
        random_seed: Optional[int] = 42
    ):
        """
        Initialize the generator.
        
        Args:
            n_timesteps: Number of time steps to generate
            sampling_rate: Sampling rate in Hz
            random_seed: Random seed for reproducibility
        """
        self.n_timesteps = n_timesteps
        self.sampling_rate = sampling_rate
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
    def _generate_base_signal(
        self,
        channel_config: Dict,
        orbit_period: float = 90 * 60  # 90 minutes in seconds
    ) -> np.ndarray:
        """
        Generate base signal with orbital patterns.
        
        Spacecraft telemetry often shows periodic patterns due to:
        - Orbital period (eclipse/sunlight cycles)
        - Thermal cycling
        - Power generation cycles
        """
        t = np.arange(self.n_timesteps) / self.sampling_rate
        
        # Base value in middle of range
        base = (channel_config["min"] + channel_config["max"]) / 2
        amplitude = (channel_config["max"] - channel_config["min"]) / 4
        
        # Orbital periodicity
        orbital_freq = 2 * np.pi / orbit_period
        signal = base + amplitude * np.sin(orbital_freq * t)
        
        # Add some secondary harmonics
        signal += 0.2 * amplitude * np.sin(2 * orbital_freq * t + np.pi/4)
        signal += 0.1 * amplitude * np.sin(3 * orbital_freq * t + np.pi/3)
        
        # Add noise
        noise = np.random.normal(0, channel_config["noise"], self.n_timesteps)
        signal += noise
        
        # Clip to valid range
        signal = np.clip(signal, channel_config["min"], channel_config["max"])
        
        return signal
    
    def _inject_spike_anomaly(
        self,
        signal: np.ndarray,
        start_idx: int,
        duration: int = 5,
        magnitude: float = 3.0
    ) -> np.ndarray:
        """Inject a spike anomaly."""
        signal = signal.copy()
        std = np.std(signal)
        direction = np.random.choice([-1, 1])
        
        spike = direction * magnitude * std * np.exp(
            -np.linspace(0, 3, duration)
        )
        
        end_idx = min(start_idx + duration, len(signal))
        signal[start_idx:end_idx] += spike[:end_idx - start_idx]
        
        return signal
    
    def _inject_drift_anomaly(
        self,
        signal: np.ndarray,
        start_idx: int,
        duration: int = 100,
        magnitude: float = 2.0
    ) -> np.ndarray:
        """Inject a gradual drift anomaly."""
        signal = signal.copy()
        std = np.std(signal)
        direction = np.random.choice([-1, 1])
        
        end_idx = min(start_idx + duration, len(signal))
        drift = direction * magnitude * std * np.linspace(
            0, 1, end_idx - start_idx
        )
        signal[start_idx:end_idx] += drift
        
        return signal
    
    def _inject_step_anomaly(
        self,
        signal: np.ndarray,
        start_idx: int,
        duration: int = 50,
        magnitude: float = 2.5
    ) -> np.ndarray:
        """Inject a step change anomaly."""
        signal = signal.copy()
        std = np.std(signal)
        direction = np.random.choice([-1, 1])
        
        end_idx = min(start_idx + duration, len(signal))
        signal[start_idx:end_idx] += direction * magnitude * std
        
        return signal
    
    def _inject_dropout_anomaly(
        self,
        signal: np.ndarray,
        start_idx: int,
        duration: int = 10
    ) -> np.ndarray:
        """Inject a signal dropout (zeros or NaN-like values)."""
        signal = signal.copy()
        end_idx = min(start_idx + duration, len(signal))
        
        # Simulate different dropout patterns
        dropout_type = np.random.choice(["zero", "constant", "noise"])
        
        if dropout_type == "zero":
            signal[start_idx:end_idx] = 0
        elif dropout_type == "constant":
            signal[start_idx:end_idx] = signal[start_idx - 1] if start_idx > 0 else 0
        else:
            signal[start_idx:end_idx] = np.random.uniform(-999, 999, end_idx - start_idx)
            
        return signal
    
    def _inject_noise_anomaly(
        self,
        signal: np.ndarray,
        start_idx: int,
        duration: int = 30,
        noise_multiplier: float = 5.0
    ) -> np.ndarray:
        """Inject increased noise/variance anomaly."""
        signal = signal.copy()
        std = np.std(signal)
        
        end_idx = min(start_idx + duration, len(signal))
        extra_noise = np.random.normal(0, noise_multiplier * std, end_idx - start_idx)
        signal[start_idx:end_idx] += extra_noise
        
        return signal
    
    def generate(
        self,
        anomaly_ratio: float = 0.05,
        channels: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic telemetry data with anomalies.
        
        Args:
            anomaly_ratio: Fraction of time steps that should contain anomalies
            channels: List of channels to generate (default: all)
            
        Returns:
            data: DataFrame with telemetry data
            labels: DataFrame with anomaly labels and metadata
        """
        if channels is None:
            channels = list(self.CHANNELS.keys())
            
        # Generate base signals
        data = {}
        for channel in channels:
            if channel in self.CHANNELS:
                data[channel] = self._generate_base_signal(self.CHANNELS[channel])
                
        # Create time index
        timestamps = pd.date_range(
            start="2024-01-01",
            periods=self.n_timesteps,
            freq=f"{int(1000/self.sampling_rate)}ms"
        )
        
        # Initialize labels
        labels = np.zeros(self.n_timesteps, dtype=int)
        anomaly_info = []
        
        # Calculate number of anomalies to inject
        n_anomalies = int(self.n_timesteps * anomaly_ratio / 20)  # ~20 timesteps per anomaly
        
        # Inject anomalies
        anomaly_methods = {
            "spike": (self._inject_spike_anomaly, {"duration": 5, "magnitude": 3.0}),
            "drift": (self._inject_drift_anomaly, {"duration": 100, "magnitude": 2.0}),
            "step": (self._inject_step_anomaly, {"duration": 50, "magnitude": 2.5}),
            "dropout": (self._inject_dropout_anomaly, {"duration": 10}),
            "noise": (self._inject_noise_anomaly, {"duration": 30, "noise_multiplier": 5.0}),
        }
        
        used_intervals = []
        
        for _ in range(n_anomalies):
            # Select random anomaly type, channel, and position
            anomaly_type = np.random.choice(list(anomaly_methods.keys()))
            channel = np.random.choice(channels)
            
            method, params = anomaly_methods[anomaly_type]
            duration = params.get("duration", 20)
            
            # Find valid start position (avoid overlapping anomalies)
            max_attempts = 100
            for _ in range(max_attempts):
                start_idx = np.random.randint(100, self.n_timesteps - duration - 100)
                
                # Check for overlap with existing anomalies
                overlap = False
                for used_start, used_end in used_intervals:
                    if not (start_idx + duration < used_start or start_idx > used_end):
                        overlap = True
                        break
                        
                if not overlap:
                    break
            else:
                continue  # Skip if can't find valid position
                
            # Inject anomaly
            data[channel] = method(data[channel], start_idx, **params)
            
            # Update labels
            end_idx = min(start_idx + duration, self.n_timesteps)
            labels[start_idx:end_idx] = 1
            used_intervals.append((start_idx, end_idx))
            
            anomaly_info.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "channel": channel,
                "type": anomaly_type,
                "timestamp": timestamps[start_idx]
            })
            
        # Create DataFrames
        df_data = pd.DataFrame(data, index=timestamps)
        df_data.index.name = "timestamp"
        
        df_labels = pd.DataFrame({
            "timestamp": timestamps,
            "is_anomaly": labels
        })
        
        # Add anomaly details
        self.anomaly_info = pd.DataFrame(anomaly_info) if anomaly_info else pd.DataFrame()
        
        return df_data, df_labels
    
    def get_anomaly_summary(self) -> pd.DataFrame:
        """Get summary of injected anomalies."""
        if hasattr(self, 'anomaly_info') and len(self.anomaly_info) > 0:
            return self.anomaly_info
        return pd.DataFrame()


def generate_demo_dataset(
    n_timesteps: int = 5000,
    anomaly_ratio: float = 0.05,
    save_path: Optional[str] = None,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to generate a demo dataset.
    
    Args:
        n_timesteps: Number of time steps
        anomaly_ratio: Fraction of anomalous time steps
        save_path: Optional path to save the data
        random_seed: Random seed
        
    Returns:
        data: Telemetry data DataFrame
        labels: Labels DataFrame
    """
    generator = SpacecraftTelemetryGenerator(
        n_timesteps=n_timesteps,
        random_seed=random_seed
    )
    
    data, labels = generator.generate(anomaly_ratio=anomaly_ratio)
    
    if save_path:
        data.to_csv(f"{save_path}/telemetry.csv")
        labels.to_csv(f"{save_path}/labels.csv", index=False)
        
        anomaly_summary = generator.get_anomaly_summary()
        if len(anomaly_summary) > 0:
            anomaly_summary.to_csv(f"{save_path}/anomaly_summary.csv", index=False)
            
    return data, labels


if __name__ == "__main__":
    # Demo
    print("Generating synthetic spacecraft telemetry...")
    data, labels = generate_demo_dataset(n_timesteps=5000, anomaly_ratio=0.05)
    
    print(f"\nData shape: {data.shape}")
    print(f"Channels: {list(data.columns)}")
    print(f"\nAnomaly distribution:")
    print(labels['is_anomaly'].value_counts())
    print(f"\nAnomaly ratio: {labels['is_anomaly'].mean():.2%}")
