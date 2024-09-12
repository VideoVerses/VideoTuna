
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification
import torch
import numpy as np

class ActPredScorer(torch.nn.Module):

    def __init__(self, model_name = "MCG-NJU/videomae-base-finetuned-kinetics", num_frames = 16, device = 'cuda', dtype=torch.float32):
        super().__init__()
        self.model = VideoMAEForVideoClassification.from_pretrained(model_name, num_frames = num_frames, torch_dtype=dtype)
        self.feature_extractor = VideoMAEFeatureExtractor.from_pretrained(model_name)
        self.device = device
        self.model.to(device)

    def get_target_class_idx(self, target_action):
        def mapping_func(x):
            if 'piano' in x:
                return 'playing piano'
            if 'guitar' in x:
                return 'playing guitar'
            if 'doughnuts' in x:
                return 'eating doughnuts'
            if 'beer' in x:
                return 'drinking beer'
            if 'badminton' in x:
                return 'playing badminton'
            if 'cello' in x:
                return 'playing cello'
            if 'scooter' in x:
                return 'riding scooter'
            if 'ballet' in x:
                return 'dancing ballet'
            if 'pancake' in x:
                return 'flipping pancake'
            if 'violin' in x:
                return 'playing violin'
            if 'wood' in x:
                return 'chopping wood'
            if 'watermelon' in x:
                return 'eating watermelon'
            if 'jogging' in x:
                return 'jogging'
            else:
                print(f"Please add your action mapping to ActPredScorer. Mapping not found for {x}")
                raise NotImplementedError
            
            
        try:
            target_class_idx = self.model.config.label2id[target_action]
        except: 
            target_class_idx = self.model.config.label2id[mapping_func(target_action)]
        return target_class_idx 

    def get_loss_and_score(self, norm_vid, target_action):
        ''' video should be a torch array of dtype float, with values from 0-1, of dimension (num_frames, height, width, 3)'''

        target_class_idx = self.get_target_class_idx(target_action)
        outputs = self.model(norm_vid, labels = torch.tensor([target_class_idx]).to(self.device))
        loss = outputs.loss
        logits = outputs.logits

        norm_logits = torch.exp(logits)/ (torch.exp(logits).sum())
        norm_logits = norm_logits.squeeze()
        
        score = norm_logits[target_class_idx]
        return loss, score, self.get_pred_class(logits)
    
    def get_pred_class(self, logits):
        predicted_class_idx = logits.argmax(-1).item()
        return self.model.config.id2label[predicted_class_idx]

def gen_rand_labels_file(labels_list, out_file, num_labels = 50):
    idxs = np.random.choice(len(labels_list), num_labels, replace = False)
    rand_labels = [labels_list[i] for i in idxs]
    rand_labels.sort()
    with open(out_file, 'w') as f:
        for line in rand_labels:
            f.write(f"{line}\n")

if __name__ == '__main__':
    # import numpy as np
    # scorer = ActPredScorer(num_frames = 7)
    # video_torch = [torch.randn((3,256,256)).clamp(0,1) for _ in range(7)]
    # encoding = scorer.feature_extractor(video_torch,  do_rescale = False, return_tensors="pt")
    # print(scorer.get_loss_and_score(video_torch))
    scorer = ActPredScorer(num_frames = 7)
    labels = scorer.model.config.id2label
    