#!/usr/bin/env python3
"""
Direct AI Model Test
Tests TensorFlow and PyTorch models directly with synthetic data
"""

import sys
import os
import numpy as np
import pandas as pd

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_tensorflow_models():
    """Test TensorFlow model creation and prediction"""
    print("=" * 60)
    print("TEST 1: TensorFlow Models Direct Test")
    print("=" * 60)
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        
        print(f"✓ TensorFlow {tf.__version__} imported successfully")
        
        # Test DNN Model
        print("\n--- Testing DNN Model ---")
        dnn_model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(20,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='softmax')  # 8 horses
        ])
        
        dnn_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Test with synthetic data
        X_test = np.random.random((10, 20))  # 10 samples, 20 features
        predictions = dnn_model.predict(X_test, verbose=0)
        print(f"✓ DNN Model prediction successful: shape {predictions.shape}")
        print(f"✓ Sample prediction: {predictions[0]}")
        
        # Test LSTM Model
        print("\n--- Testing LSTM Model ---")
        lstm_model = keras.Sequential([
            layers.LSTM(50, return_sequences=True, input_shape=(10, 20)),
            layers.Dropout(0.3),
            layers.LSTM(30),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='softmax')
        ])
        
        lstm_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Test with sequence data
        X_seq = np.random.random((5, 10, 20))  # 5 samples, 10 timesteps, 20 features
        lstm_predictions = lstm_model.predict(X_seq, verbose=0)
        print(f"✓ LSTM Model prediction successful: shape {lstm_predictions.shape}")
        print(f"✓ Sample LSTM prediction: {lstm_predictions[0]}")
        
        # Test CNN Model
        print("\n--- Testing CNN Model ---")
        cnn_model = keras.Sequential([
            layers.Reshape((20, 20, 1), input_shape=(400,)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(8, activation='softmax')
        ])
        
        cnn_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Test with image-like data
        X_cnn = np.random.random((5, 400))  # 5 samples, 400 features (20x20)
        cnn_predictions = cnn_model.predict(X_cnn, verbose=0)
        print(f"✓ CNN Model prediction successful: shape {cnn_predictions.shape}")
        print(f"✓ Sample CNN prediction: {cnn_predictions[0]}")
        
        return True
        
    except Exception as e:
        print(f"✗ TensorFlow models test failed: {e}")
        return False

def test_pytorch_models():
    """Test PyTorch model creation and prediction"""
    print("\n" + "=" * 60)
    print("TEST 2: PyTorch Models Direct Test")
    print("=" * 60)
    
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        
        print(f"✓ PyTorch {torch.__version__} imported successfully")
        
        # Test DNN Model
        print("\n--- Testing PyTorch DNN Model ---")
        class DNNModel(nn.Module):
            def __init__(self, input_size=20, hidden_sizes=[64, 32, 16], output_size=8):
                super(DNNModel, self).__init__()
                self.layers = nn.ModuleList()
                
                # Input layer
                self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
                
                # Hidden layers
                for i in range(len(hidden_sizes) - 1):
                    self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
                
                # Output layer
                self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
                self.dropout = nn.Dropout(0.3)
                
            def forward(self, x):
                for i, layer in enumerate(self.layers[:-1]):
                    x = F.relu(layer(x))
                    x = self.dropout(x)
                x = self.layers[-1](x)
                return F.softmax(x, dim=1)
        
        dnn_model = DNNModel()
        
        # Test with synthetic data
        X_test = torch.randn(10, 20)  # 10 samples, 20 features
        with torch.no_grad():
            predictions = dnn_model(X_test)
        print(f"✓ PyTorch DNN prediction successful: shape {predictions.shape}")
        print(f"✓ Sample prediction: {predictions[0]}")
        
        # Test RNN Model
        print("\n--- Testing PyTorch RNN Model ---")
        class RNNModel(nn.Module):
            def __init__(self, input_size=20, hidden_size=50, num_layers=2, output_size=8):
                super(RNNModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
                self.fc = nn.Linear(hidden_size, output_size)
                self.dropout = nn.Dropout(0.3)
                
            def forward(self, x):
                # Initialize hidden state
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                
                # Forward propagate LSTM
                out, _ = self.lstm(x, (h0, c0))
                
                # Take the last output
                out = self.dropout(out[:, -1, :])
                out = self.fc(out)
                return F.softmax(out, dim=1)
        
        rnn_model = RNNModel()
        
        # Test with sequence data
        X_seq = torch.randn(5, 10, 20)  # 5 samples, 10 timesteps, 20 features
        with torch.no_grad():
            rnn_predictions = rnn_model(X_seq)
        print(f"✓ PyTorch RNN prediction successful: shape {rnn_predictions.shape}")
        print(f"✓ Sample RNN prediction: {rnn_predictions[0]}")
        
        return True
        
    except Exception as e:
        print(f"✗ PyTorch models test failed: {e}")
        return False

def test_ensemble_prediction():
    """Test ensemble prediction combining multiple models"""
    print("\n" + "=" * 60)
    print("TEST 3: Ensemble Prediction Test")
    print("=" * 60)
    
    try:
        import tensorflow as tf
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        
        # Create simple models
        tf_model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(20,)),
            tf.keras.layers.Dense(8, activation='softmax')
        ])
        tf_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.fc1 = nn.Linear(20, 32)
                self.fc2 = nn.Linear(32, 8)
                
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.softmax(self.fc2(x), dim=1)
                return x
        
        torch_model = SimpleModel()
        
        # Generate test data
        X_test = np.random.random((5, 20))
        
        # Get predictions from both models
        tf_pred = tf_model.predict(X_test, verbose=0)
        
        X_torch = torch.tensor(X_test, dtype=torch.float32)
        with torch.no_grad():
            torch_pred = torch_model(X_torch).numpy()
        
        # Ensemble prediction (simple average)
        ensemble_pred = (tf_pred + torch_pred) / 2
        
        print(f"✓ TensorFlow predictions shape: {tf_pred.shape}")
        print(f"✓ PyTorch predictions shape: {torch_pred.shape}")
        print(f"✓ Ensemble predictions shape: {ensemble_pred.shape}")
        
        # Show sample predictions
        print(f"\nSample predictions for first horse:")
        print(f"  TensorFlow: {tf_pred[0]}")
        print(f"  PyTorch:    {torch_pred[0]}")
        print(f"  Ensemble:   {ensemble_pred[0]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Ensemble prediction test failed: {e}")
        return False

def test_model_training():
    """Test model training with synthetic data"""
    print("\n" + "=" * 60)
    print("TEST 4: Model Training Test")
    print("=" * 60)
    
    try:
        import tensorflow as tf
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        # Generate synthetic training data
        n_samples = 1000
        n_features = 20
        n_classes = 8
        
        X_train = np.random.random((n_samples, n_features))
        y_train = np.random.randint(0, n_classes, n_samples)
        
        print(f"✓ Generated synthetic training data: {X_train.shape}, {y_train.shape}")
        
        # Test TensorFlow training
        print("\n--- Testing TensorFlow Training ---")
        tf_model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(n_features,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(n_classes, activation='softmax')
        ])
        
        tf_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train for a few epochs
        history = tf_model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)
        print(f"✓ TensorFlow training completed")
        print(f"✓ Final loss: {history.history['loss'][-1]:.4f}")
        print(f"✓ Final accuracy: {history.history['accuracy'][-1]:.4f}")
        
        # Test PyTorch training
        print("\n--- Testing PyTorch Training ---")
        class TrainableModel(nn.Module):
            def __init__(self):
                super(TrainableModel, self).__init__()
                self.fc1 = nn.Linear(n_features, 32)
                self.fc2 = nn.Linear(32, 16)
                self.fc3 = nn.Linear(16, n_classes)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        
        torch_model = TrainableModel()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(torch_model.parameters())
        
        # Convert data to tensors
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.long)
        
        # Train for a few epochs
        torch_model.train()
        for epoch in range(3):
            optimizer.zero_grad()
            outputs = torch_model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        print(f"✓ PyTorch training completed")
        print(f"✓ Final loss: {loss.item():.4f}")
        
        # Test predictions after training
        torch_model.eval()
        with torch.no_grad():
            test_outputs = torch_model(X_tensor[:5])
            test_predictions = torch.softmax(test_outputs, dim=1)
        
        print(f"✓ Post-training predictions shape: {test_predictions.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model training test failed: {e}")
        return False

def main():
    """Run all direct AI model tests"""
    print("DIRECT AI MODELS TEST")
    print("=" * 60)
    
    test_results = []
    
    # Test 1: TensorFlow Models
    test_results.append(test_tensorflow_models())
    
    # Test 2: PyTorch Models
    test_results.append(test_pytorch_models())
    
    # Test 3: Ensemble Prediction
    test_results.append(test_ensemble_prediction())
    
    # Test 4: Model Training
    test_results.append(test_model_training())
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    test_names = [
        "TensorFlow Models",
        "PyTorch Models",
        "Ensemble Prediction",
        "Model Training"
    ]
    
    passed = sum(test_results)
    total = len(test_results)
    
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All AI model tests passed! TensorFlow and PyTorch are working properly.")
        print("✓ TensorFlow models (DNN, LSTM, CNN) are functional")
        print("✓ PyTorch models (DNN, RNN) are functional")
        print("✓ Ensemble prediction is working")
        print("✓ Model training is working")
    elif passed >= total * 0.75:
        print("⚠️  Most tests passed. Minor issues may need attention.")
    else:
        print("❌ Multiple test failures. Significant issues need to be resolved.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)