
The CarRacing-v2 environment relies on the Box2D physics engine. This engine is written in C++ and requires swig (Simplified Wrapper and Interface Generator) to compile the Python bindings. 

Linux (Ubuntu/Debian) - Recommended


# 1. Install system-level dependencies
```bash
sudo apt-get update
sudo apt-get install swig build-essential python3-dev
```

# 2. Install the python libraries
```bash
pip install "gymnasium[box2d]" numpy opencv-python
```
# 3. R-sync to copy training data between machines
```bash
rsync -avz --progress data/ andrewknowles@pop-os-9060xt:~/Workspace/TinyAutoJEPA/data/
```
---

Key Features of this Script:

    Multiprocessing: It spawns 16 workers (utilizing your 5950X's 32 threads comfortably) to generate data 16x faster.

    Compression: It resizes images to 64x64 and saves them as uint8. This reduces the dataset size from ~50GB to ~4GB while keeping all necessary information.

    Action Bias: Random actions in CarRacing usually result in the car spinning in place. I've added a bias helper to force the gas pedal so the car actually explores the track.