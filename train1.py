import os
import torch
import torchaudio
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchaudio
import logging
from torch.utils.data import Dataset
from itertools import permutations
from mir_eval.separation import bss_eval_sources

# Set up logging to track training progress
logging.basicConfig(filename='training_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

class WSJ0MixDataset(Dataset):
    """
    Dataset class for loading WSJ0-Mix dataset for speech separation.
    This dataset contains mixtures of two speakers and their individual clean sources.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.mixture_files = []
        self.s1_files = []
        self.s2_files = []

        # Find all mixture files and corresponding source files
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".wav") and "mix" in root:
                    mixture_path = os.path.join(root, file)
                    self.mixture_files.append(mixture_path)
                    s1_path = mixture_path.replace("mix", "s1")
                    s2_path = mixture_path.replace("mix", "s2")
                    self.s1_files.append(s1_path)
                    self.s2_files.append(s2_path)
    
    def __len__(self):
        return len(self.mixture_files)
    
    def __getitem__(self, idx):
        """
        Load audio files for mixture and individual sources at the given index.
        Returns:
            mixture_waveform: The mixed audio signal
            s1_waveform: Source 1 clean audio
            s2_waveform: Source 2 clean audio
            sample_rate: Audio sample rate
        """
        mixture_path = self.mixture_files[idx]
        s1_path = self.s1_files[idx]
        s2_path = self.s2_files[idx]

        mixture_waveform, sample_rate = torchaudio.load(mixture_path)
        s1_waveform, _ = torchaudio.load(s1_path)
        s2_waveform, _ = torchaudio.load(s2_path)
        return mixture_waveform, s1_waveform, s2_waveform, sample_rate

def transform_waveform(waveform):
    """
    Transform audio waveform to magnitude spectrogram using STFT.
    
    Args:
        waveform: Input audio waveform tensor
        
    Returns:
        magnitude: Magnitude spectrogram of the input waveform
    """
    window = torch.hann_window(512, device=waveform.device)
    stft = torch.stft(waveform, n_fft=512, hop_length=256, win_length=512, window=window, return_complex=True)
    magnitude = torch.abs(stft)
    # magnitude = (magnitude - magnitude.mean()) / magnitude.std()
    # 不要使用归一化！！！
    return magnitude

class LSTM(nn.Module):
    """
    Bidirectional LSTM model for speech separation.
    Includes residual connections to improve gradient flow.
    
    Args:
        input_dim: Dimension of input features
        hidden_dim: Number of hidden units in LSTM
        output_dim: Dimension of output features
        layers: Number of LSTM layers
    """
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.residual_fc = nn.Linear(input_dim, output_dim)
        
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_dim]
            
        Returns:
            Output tensor with residual connection
        """
        h0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)

        # Residual connection
        residual = self.residual_fc(x)
        out += residual
        return out

def pit_loss(preds, targets, mixture_waveform):
    """
    Permutation Invariant Training (PIT) loss for speech separation.
    Finds the best permutation of predictions to match targets.
    
    Args:
        preds: Model predictions (masks)
        targets: Ground truth targets
        mixture_waveform: Original mixture spectrogram
        
    Returns:
        Minimum loss across all possible permutations
    """
    batch_size, num_frames, num_frequencies, num_sources = preds.shape
    perms = list(permutations(range(num_sources)))
    min_loss = None

    for perm in perms:
        loss = 0
        for i, j in enumerate(perm):
            # [batch_size, num_frames, num_frequencies, num_sources]
            separated_audio = preds[:, :, :, i] * mixture_waveform.squeeze(1).permute(0, 2, 1)
            loss += nn.MSELoss()(separated_audio, targets[:, :, :, j])
        if min_loss is None or loss < min_loss:
            min_loss = loss
    return min_loss

def save_audio(model, dataset, device, output_dir = 'output'):
    """
    Save separated audio files for evaluation and listening.
    
    Args:
        model: Trained separation model
        dataset: Dataset containing audio samples
        device: Processing device (CPU/GPU)
        output_dir: Directory to save output audio files
    """
    mixture_waveform, s1_waveform, s2_waveform, sample_rate = dataset[0]
    original_mixture_waveform = mixture_waveform.to(device)
    mixture_waveform = transform_waveform(mixture_waveform).to(device)
    inputs = mixture_waveform.squeeze(1).permute(0, 2, 1)

    model.eval()
    with torch.no_grad():
        outputs = model(inputs).view(inputs.size(0), inputs.size(1), 257, 2)

    # Apply masks to separate sources
    mask1 = outputs[:, :, :, 0]
    mask2 = outputs[:, :, :, 1]
    separated_audio1 = mask1 * mixture_waveform.squeeze(1).permute(0, 2, 1)
    separated_audio2 = mask2 * mixture_waveform.squeeze(1).permute(0, 2, 1)
    separated_audio1 = separated_audio1.permute(0, 2, 1)
    separated_audio2 = separated_audio2.permute(0, 2, 1)

    def istft(magnitude, phase, n_fft=512, hop_length=256, win_length=512):
        """
        Inverse STFT to convert spectrogram back to waveform.
        
        Args:
            magnitude: Magnitude spectrogram
            phase: Phase information
            n_fft, hop_length, win_length: STFT parameters
            
        Returns:
            Reconstructed waveform
        """
        window = torch.hann_window(n_fft).to(magnitude.device)
        complex_spec = magnitude * torch.exp(1j * phase)
        waveform = torch.istft(complex_spec, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
        return waveform

    # Extract phase from original mixture for reconstruction
    window = torch.hann_window(512, device=mixture_waveform.device)
    stft = torch.stft(original_mixture_waveform, n_fft=512, hop_length=256, win_length=512, window=window, return_complex=True)
    phase = torch.angle(stft)

    # Reconstruct time-domain signals
    separated_waveform1 = istft(separated_audio1, phase).squeeze(1)
    separated_waveform2 = istft(separated_audio2, phase).squeeze(1)

    # Save all audio files
    os.makedirs(output_dir, exist_ok=True)
    torchaudio.save(os.path.join(output_dir, 'separated_audio1.wav'), separated_waveform1.cpu(), sample_rate)
    torchaudio.save(os.path.join(output_dir, 'separated_audio2.wav'), separated_waveform2.cpu(), sample_rate)
    torchaudio.save(os.path.join(output_dir, 'original_mixture.wav'), original_mixture_waveform.cpu(), sample_rate)
    torchaudio.save(os.path.join(output_dir, 'original_s1.wav'), s1_waveform.cpu(), sample_rate)
    torchaudio.save(os.path.join(output_dir, 'original_s2.wav'), s2_waveform.cpu(), sample_rate)
    
def evaluate(model, dataset, device):
    """
    Evaluate model performance on validation set using PIT loss.
    
    Args:
        model: Speech separation model
        dataset: Validation dataset
        device: Processing device (CPU/GPU)
        
    Returns:
        Average loss across the dataset
    """
    model.eval()
    loss = 0
    with torch.no_grad():
        for idx in range(len(dataset)):
            mixture_waveform, s1_waveform, s2_waveform, _ = val_dataset[idx]
            mixture_waveform_orignal = mixture_waveform.to(device)
            s1_waveform_orignal = s1_waveform.to(device)
            s2_waveform_orignal = s2_waveform.to(device)
            # stft
            mixture_waveform = transform_waveform(mixture_waveform).to(device)
            s1_waveform = transform_waveform(s1_waveform).to(device)
            s2_waveform = transform_waveform(s2_waveform).to(device)

            inputs = mixture_waveform.squeeze(1).permute(0, 2, 1)
            targets = torch.stack([s1_waveform.squeeze(1).permute(0, 2, 1), s2_waveform.squeeze(1).permute(0, 2, 1)], dim=3)  
            
            outputs = model(inputs).view(inputs.size(0), inputs.size(1), 257, 2)
            loss += pit_loss(outputs, targets, mixture_waveform)    
    return loss / len(dataset)

def evaluate_pit(model, dataset, device):
    """
    Evaluate model using SDR (Signal-to-Distortion Ratio), SIR (Signal-to-Interference Ratio),
    and SAR (Signal-to-Artifact Ratio) metrics.
    
    Args:
        model: Speech separation model
        dataset: Evaluation dataset
        device: Processing device (CPU/GPU)
        
    Returns:
        Average SDR, SIR, and SAR values across the dataset
    """
    model.eval()
    total_sdr = 0
    total_sir = 0
    total_sar = 0
    num_samples = 0

    with torch.no_grad():
        for idx in range(len(dataset)):
            mixture_waveform, s1_waveform, s2_waveform, sample_rate = dataset[idx]
            original_mixture_waveform = mixture_waveform.to(device)
            original_s1_waveform = s1_waveform.to(device)
            original_s2_waveform = s2_waveform.to(device)
            #stft
            mixture_waveform = transform_waveform(mixture_waveform).to(device)
            s1_waveform = transform_waveform(s1_waveform).to(device)
            s2_waveform = transform_waveform(s2_waveform).to(device)

            inputs = mixture_waveform.squeeze(1).permute(0, 2, 1)
            outputs = model(inputs).view(inputs.size(0), inputs.size(1), 257, 2)

            # Apply masks to get separated spectrograms
            mask1 = outputs[:, :, :, 0]
            mask2 = outputs[:, :, :, 1]
            separated_audio1 = mask1 * mixture_waveform.squeeze(1).permute(0, 2, 1)
            separated_audio2 = mask2 * mixture_waveform.squeeze(1).permute(0, 2, 1)
            separated_audio1 = separated_audio1.permute(0, 2, 1)
            separated_audio2 = separated_audio2.permute(0, 2, 1)

            def istft(magnitude, phase, n_fft=512, hop_length=256, win_length=512):
                window = torch.hann_window(n_fft).to(magnitude.device)
                complex_spec = magnitude * torch.exp(1j * phase)
                waveform = torch.istft(complex_spec, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
                return waveform
            
            # Get phase from original mixture for reconstruction
            window = torch.hann_window(512, device=original_mixture_waveform.device)
            stft = torch.stft(original_mixture_waveform, n_fft=512, hop_length=256, win_length=512, window=window, return_complex=True)
            phase = torch.angle(stft)

            # Convert back to time domain
            separated_waveform1 = istft(separated_audio1, phase).squeeze(1)
            separated_waveform2 = istft(separated_audio2, phase).squeeze(1)

            # Convert to numpy for evaluation
            separated_waveform1_np = separated_waveform1.cpu().numpy()
            separated_waveform2_np = separated_waveform2.cpu().numpy()
            original_s1_waveform_np = original_s1_waveform.cpu().numpy()
            original_s2_waveform_np = original_s2_waveform.cpu().numpy()

            # Handle length mismatch between separated and original sources
            if separated_waveform1_np.shape[1] > original_s1_waveform_np.shape[1]:
                original_s1_waveform_np = np.pad(original_s1_waveform.cpu().numpy(), ((0, 0), (0, separated_waveform1_np.shape[1] - original_s1_waveform.shape[1])), 'constant')
                original_s2_waveform_np = np.pad(original_s2_waveform.cpu().numpy(), ((0, 0), (0, separated_waveform1_np.shape[1] - original_s2_waveform.shape[1])), 'constant')
            else:
                separated_waveform1_np = np.pad(separated_waveform1_np, ((0, 0), (0, original_s1_waveform.shape[1] - separated_waveform1_np.shape[1])), 'constant')
                separated_waveform2_np = np.pad(separated_waveform2_np, ((0, 0), (0, original_s2_waveform.shape[1] - separated_waveform2_np.shape[1])), 'constant')

            # Calculate SDR, SIR, SAR metrics using mir_eval
            sdr1, sir1, sar1, _ = bss_eval_sources(original_s1_waveform_np, separated_waveform1_np) 
            sdr2, sir2, sar2, _ = bss_eval_sources(original_s2_waveform_np, separated_waveform2_np) 
            sdr = (sdr1 + sdr2) / 2
            sir = (sir1 + sir2) / 2
            sar = (sar1 + sar2) / 2

            total_sdr += sdr.mean()
            total_sir += sir.mean()
            total_sar += sar.mean()
            num_samples += 1    

    # Calculate average metrics
    avg_sdr = total_sdr / num_samples
    avg_sir = total_sir / num_samples
    avg_sar = total_sar / num_samples
    return avg_sdr, avg_sir, avg_sar

# Dataset paths and initialization
dataset_path = "/data/lib/WSJ0_MIX/2speakers/wav8k/min/"
train_dataset = WSJ0MixDataset(root_dir=os.path.join(dataset_path, 'tr'))
val_dataset = WSJ0MixDataset(root_dir=os.path.join(dataset_path, 'cv'))
test_dataset = WSJ0MixDataset(root_dir=os.path.join(dataset_path, 'tt'))

# Model configuration
input_dim = 257  # STFT output dimension
hidden_dim = 512
output_dim = 257 * 2 # Output dimension for two masks (one per speaker)
num_layers = 3

# Set up device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("cuda is available")

# Initialize model and optimizer
model = LSTM(input_dim, hidden_dim, output_dim, num_layers)
model = nn.DataParallel(model).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

def save_checkpoint(epoch, model, optimizer, loss, filename='checkpoint1.pth.tar'):
    """
    Save model checkpoint to resume training later.
    
    Args:
        epoch: Current epoch number
        model: Model state
        optimizer: Optimizer state
        loss: Current loss value
        filename: File to save checkpoint
    """
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(state, filename)

def sdr_loss(preds, targets, mixture_waveform):
    """
    SDR-based loss function with permutation invariance.
    Optimizes directly for Signal-to-Distortion Ratio.
    
    Args:
        preds: Model predictions
        targets: Target sources
        mixture_waveform: Original mixture spectrogram
        
    Returns:
        Minimum negative SDR loss across all permutations
    """
    batch_size, num_frames, num_frequencies, num_sources = preds.shape

    perms = list(permutations(range(num_sources)))
    min_loss = None

    for perm in perms:
        perm_loss = 0
        for i, j in enumerate(perm):
            separated_audio = preds[:, :, :, i] * mixture_waveform.squeeze(1).permute(0, 2, 1)
            reference_audio = targets[:, :, :, j]

            # Calculate SDR
            noise = reference_audio - separated_audio
            sdr = 10 * torch.log10(torch.sum(reference_audio ** 2, dim=(1, 2)) / torch.sum(noise ** 2, dim=(1, 2)))

            # Loss is negative SDR (we want to maximize SDR)
            perm_loss += -sdr

        if min_loss is None or perm_loss < min_loss:
            min_loss = perm_loss

    return min_loss / num_sources

# Training loop
val_losses = []
sdr_values = []
epochs = 8
for epoch in range(epochs):  
    model.train()
    indices = torch.randperm(len(train_dataset))
    for idx in indices:
        # Load and preprocess data
        mixture_waveform, s1_waveform, s2_waveform, _ = train_dataset[idx]
        mixture_waveform_orignal = mixture_waveform.to(device)
        s1_waveform_orignal = s1_waveform.to(device)
        s2_waveform_orignal = s2_waveform.to(device)
        # Apply STFT
        mixture_waveform = transform_waveform(mixture_waveform).to(device)
        s1_waveform = transform_waveform(s1_waveform).to(device)
        s2_waveform = transform_waveform(s2_waveform).to(device)

        # Prepare inputs and targets
        inputs = mixture_waveform.squeeze(1).permute(0, 2, 1)
        targets = torch.stack([s1_waveform.squeeze(1).permute(0, 2, 1), s2_waveform.squeeze(1).permute(0, 2, 1)], dim=3)  
        outputs = model(inputs).view(inputs.size(0), inputs.size(1), 257, 2)
        
        # Calculate loss (using SDR loss instead of PIT MSE loss)
        # loss = pit_loss(outputs, targets, mixture_waveform)
        loss = sdr_loss(outputs, targets, mixture_waveform)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if idx % 1000 == 0:
            print(f"Epoch [{epoch+1}/10], Sample [{idx+1}/{len(train_dataset)}], Loss: {loss.item()}")
            logging.info(f"Epoch [{epoch+1}/10], Sample [{idx+1}/{len(train_dataset)}], Loss: {loss.item()}")
    
    # Evaluate on validation set
    val_loss = evaluate(model, val_dataset, device)
    val_losses.append(val_loss.cpu().item())
    print(f"Epoch [{epoch+1}/10], Validation Loss: {val_loss}")
    logging.info(f"Epoch [{epoch+1}/10], Validation Loss: {val_loss}")

    # Calculate and record separation metrics
    avg_sdr, avg_sir, avg_sar = evaluate_pit(model, val_dataset, device)
    sdr_values.append(avg_sdr)
    logging.info(f"Epoch [{epoch+1}/10], Validation SDR: {avg_sdr:.4f}, SIR: {avg_sir:.4f}, SAR: {avg_sar:.4f}")

    with open('outcome.txt', 'a') as f:
        f.write(f"Epoch [{epoch+1}/10], Validation SDR: {avg_sdr:.4f}, SIR: {avg_sir:.4f}, SAR: {avg_sar:.4f}\n")

# Final evaluation on validation and test sets
avg_sdr, avg_sir, avg_sar = evaluate_pit(model, val_dataset, device)
test_sdr, test_sir, test_sar = evaluate_pit(model, test_dataset, device)
with open('outcome.txt', 'a') as f:
        f.write(f"TOTAL: Validation SDR: {avg_sdr:.4f}, SIR: {avg_sir:.4f}, SAR: {avg_sar:.4f}\n")
        f.write(f"TOTAL: Test SDR: {test_sdr:.4f}, SIR: {test_sir:.4f}, SAR: {test_sar:.4f}\n")


def load_checkpoint(filename='checkpoint1.pth.tar'):
    """
    Load model checkpoint to resume training or for inference.
    
    Args:
        filename: Checkpoint file to load
        
    Returns:
        model: Loaded model
        optimizer: Loaded optimizer state
        epoch: Epoch number from checkpoint
        loss: Loss value from checkpoint
    """
    checkpoint = torch.load(filename)
    model = LSTM(input_dim, hidden_dim, output_dim, num_layers).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


# Uncomment to load a saved model
# model, optimizer, start_epoch, loss = load_checkpoint()
# model.eval()  

# Save example separated audio
save_audio(model, test_dataset, device)

# Plot validation loss over epochs
plt.figure()
plt.plot(range(1, epochs + 1), val_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss per Epoch')
plt.savefig('validation_loss.png')
plt.show()

# Plot SDR over epochs
plt.plot(range(1, epochs + 1), sdr_values, marker='o')
plt.xlabel('Epoch')
plt.ylabel('SDR')
plt.title('SDR per Epoch')
plt.grid(True)
plt.savefig('sdr_per_epoch.png')
plt.show()

