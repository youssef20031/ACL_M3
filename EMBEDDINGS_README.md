# Embeddings - Memory-Safe Build Process

## Problem
- Railway has only 512 MB RAM - not enough to build embeddings
- Building embeddings locally crashes the PC due to memory usage

## Solution: Pre-build Embeddings Offline

### Option 1: Build on a Different Machine (Recommended)
Use a machine with more RAM (at least 4 GB recommended):

```bash
# Install dependencies (if not already)
pip install psutil

# Build MiniLM embeddings (faster, smaller)
python build_embeddings_offline.py minilm

# Or build MPNet embeddings (higher quality, more memory)
python build_embeddings_offline.py mpnet
```

The script will:
- Monitor memory usage
- Use small batch sizes (8) to prevent crashes
- Save embeddings to `embeddings/prebuilt/minilm_embeddings.pkl`
- Show progress and memory stats

### Option 2: Use Google Colab (Free, Cloud-based)
1. Upload the project to Google Drive
2. Open Google Colab (colab.research.google.com)
3. Run the build script in Colab (free 12 GB RAM)
4. Download the generated `.pkl` file

### Option 3: Build with WSL2 Memory Limits
If using Windows with WSL2, limit memory:

```bash
# Create/edit ~/.wslconfig
[wsl2]
memory=2GB

# Then build with even smaller batches
# (modify build_embeddings_offline.py batch_size to 4)
```

## Deployment

Once built:

```bash
# 1. Commit the prebuilt embeddings
git add embeddings/prebuilt/*.pkl
git commit -m "Add prebuilt embeddings for Railway deployment"
git push origin main

# 2. Railway will automatically:
#    - Load prebuilt embeddings on startup
#    - Skip the memory-intensive build process
#    - Serve embeddings from disk
```

## File Sizes
- MiniLM embeddings: ~15-25 MB
- MPNet embeddings: ~30-50 MB

## Memory Usage
**Building:**
- MiniLM: ~2-3 GB RAM
- MPNet: ~4-6 GB RAM

**Loading (Railway):**
- MiniLM: ~50-100 MB RAM
- MPNet: ~100-200 MB RAM

## Troubleshooting

### PC Still Crashes
- Close all other applications
- Reduce batch_size in `build_embeddings_offline.py` to 4 or even 2
- Try Option 2 (Google Colab) instead

### Railway Still Shows 502
- Check if `.pkl` files were committed and pushed
- Check Railway logs: `Loaded X prebuilt embeddings`
- If not loading, check file path: `embeddings/prebuilt/minilm_embeddings.pkl`

### Embeddings Not Found
The API will try to load from: `embeddings/prebuilt/{model}_embeddings.pkl`
Make sure the file exists in that exact path.
