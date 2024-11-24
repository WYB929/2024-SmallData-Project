import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchaudio
import torchaudio.transforms as T

class AnimalDataset(Dataset):
    def __init__(self, root_dir, image_size=(224, 224), target_audio_length=5.0, target_sample_rate=22050, transform=None):
        """
        Custom Dataset for animal images and audio.

        Args:
            root_dir (str): Root directory of the dataset (e.g., 'training_dataset').
            image_size (tuple): Target size for resizing images (width, height).
            target_audio_length (float): Target length for audio clips in seconds.
            target_sample_rate (int): Target sample rate for audio.
            transform (torchvision.transforms): Custom transformations for images.
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.target_audio_length = target_audio_length
        self.target_sample_rate = target_sample_rate
        self.transform = transform
        self.image_paths = []  # List to store image file paths
        self.audio_paths = []  # List to store audio file paths
        self.labels = []       # Corresponding labels (category indices)
        self.class_to_idx = {} # Mapping from class name to index

        # Load the dataset
        self._load_dataset()

    def _load_dataset(self):
        # Traverse the root directory
        for idx, class_name in enumerate(sorted(os.listdir(self.root_dir))):
            class_path = self.root_dir / class_name
            if not class_path.is_dir() or class_name.startswith('.') or class_name.startswith('__'):
                continue
            
            # Map class names to indices
            self.class_to_idx[class_name] = idx

            # Process image and audio files
            image_folder = class_path / 'images'
            sound_folder = class_path / 'sound'

            if image_folder.exists():
                for image_name in os.listdir(image_folder):
                    image_path = image_folder / image_name
                    if image_name.endswith(('.jpg', '.png', '.jpeg')):  # Valid image files
                        self.image_paths.append(image_path)
                        self.labels.append(idx)

            if sound_folder.exists():
                for audio_name in os.listdir(sound_folder):
                    audio_path = sound_folder / audio_name
                    if audio_name.endswith('.wav'):  # Valid audio files
                        self.audio_paths.append(audio_path)
                        self.labels.append(idx)

    def __len__(self):
        # Return the total number of samples
        return len(self.image_paths)

    def process_audio(self, audio_path):
        # Load the audio file
        audio, original_sample_rate = torchaudio.load(audio_path)

        # Convert stereo to mono if necessary
        if audio.size(0) > 1:  # Check if audio has more than one channel
            audio = torch.mean(audio, dim=0, keepdim=True)

        # Resample to target sample rate if needed
        if original_sample_rate != self.target_sample_rate:
            resampler = T.Resample(orig_freq=original_sample_rate, new_freq=self.target_sample_rate)
            audio = resampler(audio)

        # Calculate target number of samples
        target_samples = int(self.target_audio_length * self.target_sample_rate)

        # Trim or pad the audio to match the target length
        if audio.size(1) > target_samples:
            # Trim the audio
            audio = audio[:, :target_samples]
        else:
            # Pad with zeros to the target length
            padding = target_samples - audio.size(1)
            audio = torch.nn.functional.pad(audio, (0, padding))

        # Normalize the audio to range [-1, 1]
        audio = audio / torch.abs(audio).max()

        return audio

    def __getitem__(self, idx):
        # Get label
        label = self.labels[idx]
        # Process image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")  # Ensure 3-channel RGB
        if self.transform:
            image = self.transform(image)
        else:
            # Default image transformation
            default_transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor()
            ])
            image = default_transform(image)

        # Process audio
        audio_path = self.audio_paths[idx]
        audio = self.process_audio(audio_path)

        return image, audio, label