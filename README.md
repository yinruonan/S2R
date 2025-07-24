# S2RFault
Testing models and scripts for paper **Synthetic to Real: Unsupervised Domain Adaptation for 3D Fault Detection**

**ABSTRACT:**
Fault detection is a crucial step in seismic interpretation, which can be regarded as a segmentation task in computer vision. Existing data-driven 3D fault detection approaches rely on pixel-level annotations of synthetic data. However, due to the domain shift between synthetic and field data, these methods always have limitations when generalizing to unseen seismic domains. Given the labor-intensive nature of 3D fault annotation, there is significant interest in developing algorithms that can adapt synthetic annotations to field data. Motivated by the above challenges, we proposed an unsupervised domain adaption method for 3D fault detection, named S2R. S2R is primarily composed of a Domain Invariant Learning (DIL) module and a Multi-level Perturbation Learning (MPL) module. DIL consists of a domain invariant encoder and a domain classifier. It employs pixel-level adversarial learning to encourage the encoder to learn domain invariant representations from synthetic and field data. MPL contains a teacher model and several perturbation decoders. It performs class-level confidence pseudo label selection to learn different perturbations in the latent space, enabling the exploration of field data. DIL and MPL mutually promote and constrain each other during training, thereby improving the model's performance across different field data. Through comparisons with the state-of-the-art (SOTA) methods, we demonstrate the practicality of S2R in various experimental settings, including intro-database and cross-database scenarios. Experimental results indicate that S2R outperforms comparative methods in fault detection accuracy and visual quality. S2R provides a promising approach to incorporate unlabeled field data into training and enhance model generalization.

# Download the trained models and datasets.
Google Drive: https://drive.google.com/drive/folders/1wGi8R2BKnmzdgg1RYS6FjSBZtlr8QRb2?usp=share_link

# Results
## Netherland offshore F3
<div align=center><img src="imgs/F3.jpg" width="600" alt="F3 Results"/><br/></div>

## Kerry-3D
<div align=center><img src="imgs/Kerry.jpg" width="500" alt="Kerry Results"/><br/></div>

## Clyde
<div align=center><img src="imgs/Clyde.jpg" width="400" alt="Clyde Results"/><br/></div>

## Opunake-3D
<div align=center><img src="imgs/Opunake.jpg" width="400" alt="Kerry Results"/><br/></div>

# Quick Start
Install all the requirements.  

Example of inference on Netherland offshore F3 data.  

Download F3 from the shared link, and put it into 'data'.  

Download the checkpoints from the shared link, and put them into 'model'.  

run command: python inference.py --model ./models/s2r_it_20k.pt --data ./data/F3.npy --save-filename F3_result_s2r --show

# E-mail:  
b22070022@s.upc.edu.cn
