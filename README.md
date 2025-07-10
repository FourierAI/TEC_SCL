# Text Enhanced Curriculum Supervised Contrastive Learning for Food Image Recognition

Food image recognition faces significant challenges due to high intra-class
variations, subtle inter-class distinctions, and biased hierarchical taxonomies.
Traditional methods fail to effectively capture fine-grained culinary semantics, while prevailing contrastive learning frameworks exhibit limited adaptability to the dynamic evolution of feature representations during progressive
training. To address these limitations, we propose Text Enhanced Curriculum Supervised Contrastive Learning (TEC-SCL), a novel multimodal learning framework which synergizes Vision-Language Model (VLM)-generated
semantic descriptions with visual features through cross-modal attention fusion. In addition, we introduce a curriculum-based scheduler to dynamically
optimize contrastive pairs by prioritizing hard negatives. Extensive experiments conducted on ETH Food-101, ISIA Food-500, and UEC-Food 256
datasets demonstrate that our method achieves state-of-the-art performance,
obtaining the greatest Top-1 accuracy for fine-grained retrieval. The framework bridges the gap between generic vision models and domain-specific food
image recognition, offering significant potential for intelligent food systems.

## Highlights
- A text-enhanced contrastive learning framework is proposed for finegrained food image recognition, which integrates VLM-generated semantic descriptions with visual features from images.
- A cross-modal attention fusion technique is proposed to align visual
features with textual embeddings, enhancing the discriminative performance of the model.
- A curriculum-based scheduler is introduced for hard negative mining,
which adaptively optimizes contrastive pairs during training.
- SOTA accuracy is achieved on the Food-101, Food-500, and Food-
256 benchmarks, alone with the release of the 532K image-text pairs
dataset.

## Installation
1. Clone this repository:
   ```bash
   git clone <repo-url>
   cd TEC_SCL
   ```
2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   (If `requirements.txt` is missing, install PyTorch, torchvision, and other dependencies as needed.)

## Usage
- **Training:**
  ```bash
  python main_supcon.py --config <config-file>
  ```
- **Custom Datasets:**
  Place your datasets in the `datasets/` directory and update the data loading logic in `dataload.py` as needed.
- **Text Files:**
  Text descriptions for classes are in the `text/` directory.
