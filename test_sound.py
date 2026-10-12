import winsound
import os

# Check 1: Do the WAV files actually exist?
warning_path = r"C:\Windows\Media\Windows Notify.wav"
critical_path = r"C:\Windows\Media\Windows Critical Stop.wav"

print(f"Warning file exists: {os.path.exists(warning_path)}")
print(f"Critical file exists: {os.path.exists(critical_path)}")

# Check 2: Play them synchronously (blocking)
print("\nPlaying WARNING sound...")
try:
    winsound.PlaySound(warning_path, winsound.SND_FILENAME)
    print("Warning sound played.")
except Exception as e:
    print(f"Warning failed: {e}")

# Check 3: Play critical
print("\nPlaying CRITICAL sound...")
try:
    winsound.PlaySound(critical_path, winsound.SND_FILENAME)
    print("Critical sound played.")
except Exception as e:
    print(f"Critical failed: {e}")

# Check 4: Fallback system beep
print("\nPlaying fallback system beep...")
winsound.MessageBeep(winsound.MB_ICONHAND)
print("Done.")