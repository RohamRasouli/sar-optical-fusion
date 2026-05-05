import json, os

nb = {
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.10.0"}
 },
 "cells": [
  {
   "cell_type": "code", "execution_count": None, "id": "c01", "metadata": {}, "outputs": [],
   "source": [
    "!git clone https://github.com/RohamRasouli/sar-optical-fusion.git\n",
    "!pip install rasterio -q\n",
    "!pip install -e sar-optical-fusion -q\n"
   ]
  },
  {
   "cell_type": "code", "execution_count": None, "id": "c02", "metadata": {}, "outputs": [],
   "source": [
    "EPOCHS = 8\n"
   ]
  },
  {
   "cell_type": "code", "execution_count": None, "id": "c03", "metadata": {}, "outputs": [],
   "source": [
    "!python -m src.train \\\n",
    "  --config sar-optical-fusion/configs/kaggle_p100.yaml \\\n",
    "  --data_root /kaggle/input/m4-sar/M4-SAR/M4-SAR \\\n",
    "  --output /kaggle/working/runs \\\n",
    "  --epochs {EPOCHS}\n"
   ]
  }
 ]
}

out = os.path.join(os.path.dirname(__file__), 'kaggle_train.ipynb')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f'OK: {out}')
