# QPCA-DFF
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Concatenate, Dropout, Attention
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
# Load your stock dataset
df = pd.read_csv("/content/HDB.csv", parse_dates=["Date"])

# Drop any categorical columns if they exist
df = df.drop(columns=["Close"], errors='ignore')  # Modify based on your dataset

# Impute missing values
imputer = KNNImputer(n_neighbors=2)
data_imputed = imputer.fit_transform(df.drop(columns=['Date']))

# Normalize
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_imputed)

def apply_qpca(data, time_splits=2, feature_splits=2, components=1):
    T, F = data.shape
    time_step = T // time_splits
    feat_step = F // feature_splits
    quadrant_outputs = []

    for i in range(time_splits):
        for j in range(feature_splits):
            block = data[i*time_step:(i+1)*time_step, j*feat_step:(j+1)*feat_step]
            pca = PCA(n_components=components)
            reduced = pca.fit_transform(block)
            quadrant_outputs.append(reduced)

    return np.concatenate(quadrant_outputs, axis=1)

qpca_output = apply_qpca(data_scaled)

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size, 0])  # Predict Close price (assumed to be the first)
    return np.array(X), np.array(y)

window_size = 5
X_seq, y_seq = create_sequences(qpca_output, window_size)

inp = Input(shape=(window_size, X_seq.shape[2]))
lstm_out = LSTM(64, return_sequences=True)(inp)
bilstm_out = Bidirectional(LSTM(32, return_sequences=True))(inp)
attention = Attention()([lstm_out, bilstm_out])
combined = Concatenate()([lstm_out[:, -1], bilstm_out[:, -1], attention[:, -1]])
dense1 = Dense(64, activation='relu')(combined)
dropout = Dropout(0.2)(dense1)
output = Dense(1)(dropout)

model = Model(inputs=inp, outputs=output)
model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
model.summary()

# Input
inp = Input(shape=(window_size, X_seq.shape[2]))

# LSTM
lstm_out = LSTM(64, return_sequences=True)(inp) # Changed units to 64 to match Bi-LSTM

# Bi-LSTM
bilstm_out = Bidirectional(LSTM(32, return_sequences=True))(inp)

# Attention Layer
attention = Attention()([lstm_out, bilstm_out]) # Now, both inputs have the same hidden size (64)

# Combine outputs
combined = Concatenate()([lstm_out[:, -1], bilstm_out[:, -1], attention[:, -1]])

# Dense layers
dense1 = Dense(64, activation='relu')(combined)
dropout = Dropout(0.2)(dense1)
output = Dense(1)(dropout)

# Build model
model = Model(inputs=inp, outputs=output)
model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])

model.summary()


model.fit(X_seq, y_seq, epochs=50, batch_size=4, validation_split=0.2)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Prediction
y_pred = model.predict(X_seq)

# Actual vs Predicted
plt.figure(figsize=(10, 5))
plt.plot(y_seq, label="Actual", color='blue', linewidth=2)
plt.plot(y_pred, label="Predicted", color='red', linewidth=2)
plt.title("Stock Close Price Prediction", fontweight='bold', fontsize=14)
plt.xlabel("Time Step", fontweight='bold', fontsize=12)
plt.ylabel("Close Price", fontweight='bold', fontsize=12)
plt.legend(fontsize=10, loc='upper left', prop={'weight':'bold'})
plt.grid(True, linewidth=1.2)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.tight_layout()
plt.show()

# Residuals
residuals = y_seq - y_pred.flatten()
plt.figure(figsize=(10, 4))
plt.plot(residuals, label='Residuals', color='purple', linewidth=2)
plt.axhline(0, linestyle='--', color='black', linewidth=1.5)
plt.title("Prediction Residuals-HDB", fontweight='bold', fontsize=14)
plt.xlabel("Time Step", fontweight='bold', fontsize=12)
plt.ylabel("Residual", fontweight='bold', fontsize=12)
plt.legend(fontsize=10, prop={'weight':'bold'})
plt.grid(True, linewidth=1.2)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.tight_layout()
plt.show()

# Histogram of Residuals
plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=15, color='gray', edgecolor='black')
plt.title('Histogram of Prediction Errors -HDB', fontweight='bold', fontsize=14)
plt.xlabel('Error', fontweight='bold', fontsize=12)
plt.ylabel('Frequency', fontweight='bold', fontsize=12)
plt.grid(True, linewidth=1.2)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.tight_layout()
plt.show()

# Metrics
mae = mean_absolute_error(y_seq, y_pred)
mse = mean_squared_error(y_seq, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_seq, y_pred)

print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.drop(columns=['Date']).corr(), annot=True, cmap='coolwarm',
            annot_kws={'weight': 'bold'}, cbar_kws={'label': 'Correlation Strength'})
plt.title('Feature Correlation Heatmap- HDB', fontweight='bold', fontsize=14)
plt.xticks(fontweight='bold', rotation=45)
plt.yticks(fontweight='bold', rotation=0)
plt.tight_layout()
plt.show()

# Cumulative PCA Variance
scaler_std = StandardScaler()
data_std = scaler_std.fit_transform(df.drop(columns=['Date']))
pca = PCA()
pca.fit(data_std)

plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linewidth=2)
plt.title('Cumulative PCA Variance-HDB', fontweight='bold', fontsize=14)
plt.xlabel('Components', fontweight='bold', fontsize=12)
plt.ylabel('Variance', fontweight='bold', fontsize=12)
plt.grid(True, linewidth=1.2)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.tight_layout()
plt.show()

# QPCA Features
plt.figure(figsize=(10, 4))
for i in range(qpca_output.shape[1]):
    plt.plot(qpca_output[:, i], label=f'QPCA-{i+1}', linewidth=2)
plt.title('QPCA Transformed Features-HDB', fontweight='bold', fontsize=14)
plt.xlabel('Time Step', fontweight='bold', fontsize=12)
plt.ylabel('Value', fontweight='bold', fontsize=12)
plt.legend(fontsize=10, loc='upper right', prop={'weight':'bold'})
plt.grid(True, linewidth=1.2)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.tight_layout()
plt.show()


