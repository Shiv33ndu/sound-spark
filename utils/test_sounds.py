from src.orchestrator import Orchestrator
import numpy as np, soundfile as sf, os
OUTDIR = "tests/sample_audio"
os.makedirs(OUTDIR, exist_ok=True)

sr = 22050
duration = 2.0  # seconds

def write_wav(arr, path):
    sf.write(path, arr, sr)
    print("Wrote", path)

# 1. clean sub sine (sub_bass.wav)
t = np.linspace(0, duration, int(sr*duration), endpoint=False)
sub = 0.5 * np.sin(2*np.pi*55*t)  # 55Hz sub
write_wav(sub, os.path.join(OUTDIR, "sub_bass.wav"))

# 2. gritty "reese-ish" bass (detune two saws + bit crush-ish)
f1 = 100
saw1 = 0.25 * (2*(t*f1 - np.floor(0.5 + t*f1)))
saw2 = 0.25 * (2*(t*(f1*1.01) - np.floor(0.5 + t*(f1*1.01))))
gritty = saw1 + saw2
# simple soft clip
gritty = np.tanh(gritty * 3.0)
write_wav(gritty, os.path.join(OUTDIR, "gritty_bass.wav"))

# 3. warm pad (filtered noise + slow envelope)
noise = np.random.normal(0, 0.2, size=t.shape)
env = np.linspace(0,1,t.size)**0.6
pad = np.convolve(noise*env, np.ones(500)/500, mode='same')
write_wav(pad, os.path.join(OUTDIR, "warm_pad.wav"))

# 4. pluck (short percussive pluck)
pluck = np.sin(2*np.pi*440*t) * np.exp(-6*t)
write_wav(pluck, os.path.join(OUTDIR, "pluck.wav"))

# 5. click/hit (percussive transient)
hit = np.zeros_like(t)
hit[0:200] = np.linspace(1,0,200)
write_wav(hit, os.path.join(OUTDIR, "hit.wav"))

# 6. vocal-chop-like (granular short noisy bursts)
vc = np.zeros_like(t)
for i in range(6):
    start = int(i * sr * 0.3)
    end = start + 300
    if end < len(vc):
        vc[start:end] += np.random.normal(0, 0.6, size=(end-start))*np.hanning(end-start)
write_wav(vc, os.path.join(OUTDIR, "vocal_chop.wav"))

print("Sample files:", os.listdir(OUTDIR))