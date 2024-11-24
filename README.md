# 2024-SmallData-Project
### TODO:
* First Stage: 
    * Create test set with sound and images
    * Create pure audio set for training audio-only model
    * Create pure image set for training image only model
    * Implement image+audio model(CNN, Contrastive, Generative), image-only model, audio-only model. 
    * Training and evaluation of all the models (accuracy, t-SNE)
* Second Stage: 
    * Training model with different amount of data(5 images/audios per class, 10 images/audios per class, 20 images/audios per class)
    * Evaluate scalability our model trained with audio + image
* Third Stage: 
    * Add noise for both images and audios:  Gaussian Noise with different levels
    * Evaluate robustness of models training with noise (performance drop?)

## Stats of Dataset(image + audio pair):
### Train:
| Animal   | Number |
|----------|--------|
| cat      | 15     |
| chicken  | 15     |
| cow      | 15     |
| dog      | 15     |
| donkey   | 15     |
| frog     | 15     |
| lion     | 15     |
| monkey   | 15     |
| sheep    | 15     |

### Test: 
| Animal   | Number |
|----------|--------|
| cat      | 40     |
| chicken  | 15     |
| cow      | 45     |
| dog      | 40     |
| donkey   | 10     |
| frog     | 20     |
| lion     | 30     |
| monkey   | 10     |
| sheep    | 25     |
