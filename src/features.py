import librosa
import numpy as np

def extract_basic_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_var = np.var(mfcc, axis=1)

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # Root Mean Square Energy
    rms = np.mean(librosa.feature.rms(y=y))

    features = np.concatenate([
        mfcc_mean,
        mfcc_var,
        [zcr, rms]
    ])

    return features


if __name__ == "__main__":
    human_features = extract_basic_features("test_audio/sample.mp3")
    ai_features = extract_basic_features("test_audio/ai_sample.mp3")

    print("Human features (first 5):", human_features[:5])
    print("AI features (first 5):", ai_features[:5])

    print("\nMean absolute difference:",
          np.mean(np.abs(human_features - ai_features)))

