import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_heatmaps(name, dl, maps, labels, viz_dir: Path):
    """Exports localization overlays for anomalies."""
    _MEAN, _STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    idx = np.where(labels == 1)[0]
    if len(idx) == 0: return
    
    viz_dir.mkdir(parents=True, exist_ok=True)
    i = idx[0] # Pick the first defect
    img, _ = dl.dataset[i]
    img = np.clip(img.permute(1,2,0).numpy() * _STD + _MEAN, 0, 1)
    
    hmap = cv2.applyColorMap((cv2.resize(maps[i], (224, 224)) * 255 / maps[i].max()).astype(np.uint8), cv2.COLORMAP_JET)
    hmap = cv2.cvtColor(hmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = cv2.addWeighted((img*255).astype(np.uint8), 0.5, (hmap*255).astype(np.uint8), 0.5, 0)
    
    plt.imsave(str(viz_dir / f"{name}_heatmap.png"), overlay)
