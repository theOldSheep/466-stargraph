# 466-stargraph
Stargraph models for cmput 466

# Cmd to start venv (with working directory under the parent folder of this .MD)
Set-ExecutionPolicy Unrestricted -Scope Process
.\venv\Scripts\activate.ps1

# Env setup (after venv activates)
pip3 install -r requirements.txt
(For pytorch, look up cuda version under https://pytorch.org/)

# Data preparation
python .\datagen.py --pl 5 --np 5 --ns 2000000