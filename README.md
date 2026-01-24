# Dynamic-Context-Learning-Action
# Dynamic Context Learning using Multiple Visual Scanpaths for Action Classification in Still Images

## Introduction
Source code for the paper:  
**"Dynamic Context Learning using Multiple Visual Scanpaths for Action Classification in Still Images"**.

\href{Paper link}{https://www.sciencedirect.com/science/article/pii/S1568494625018344}

Humans consistently recognize actions in images through visual observation by adeptly attending to the entities involved and identifying their contextual relationships. 
Drawing inspiration from this phenomenon, we propose a network that utilizes visual scanpaths to perform action classification in images. Specifically, the network consists of a novel dynamic context module (DCM) that implements human-like understanding by employing a scanpath-driven long short-term memory (LSTM) for enriched modeling of the dynamic interactions among visual entities involved in actions. The scanpaths on an image are generated using a human visual scanpath predictor, and our approach also includes the detection of object instances in an image and the extraction of their features. These features are subsequently enhanced by the DCM and then employed to perform the action classification. 
Extensive experiments demonstrate that integrating visual scanpaths improves action classification accuracy, achieving state-of-the-art results across different datasets. This reflects the effectiveness of our dynamic context learning strategy to perform human-like assessment for action classification in still images. This is corroborated by a thorough ablation study, which also establishes the contributions of the other model components.

---

<p align="center">
  <img src="assets/dynamic-context-network-subpage/Graphical_Abstract.jpg" width="800">
</p>

---

## Installation
This project is developed using **Python 3.9.7** and Pytorch 1.11.0.

### Python Packages
```txt
numpy
torch
opencv-python
matplotlib
tqdm
scipy
