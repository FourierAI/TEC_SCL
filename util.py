from __future__ import print_function
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import math
import numpy as np
import torch
import torch.optim as optim
from transformers import AutoProcessor, CLIPModel
from torchvision.datasets import Food101

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def get_text_embeddings(categories, model, processor):
    """Generate CLIP text embeddings using template 'a photo of a {}'"""
    # Create text inputs using template
    text_inputs = [f"a photo of a {category}" for category in categories]
    
    # Process text through CLIP
    inputs = processor(text=text_inputs, return_tensors="pt", padding=True).to('cuda')
    text_features = model.get_text_features(**inputs)
    
    # Normalize embeddings
    text_embeddings = text_features / text_features.norm(dim=-1, keepdim=True).cuda()
    
    return text_embeddings

def get_food101_categories():
    """Get all Food101 category names"""
    # Initialize Food101 dataset just to get the classes
    dataset = Food101(root='./datasets', split='train', download=True)
    # Food101 classes are already in a clean format
    categories = dataset.classes
    return categories

def get_food101_text_embeddings():
    """Get CLIP text embeddings for all Food101 categories"""
    # Load CLIP model and processor
    with torch.no_grad():
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to('cuda')
        processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    # Get Food101 categories
    categories = get_food101_categories()
    
    # Get text embeddings
    text_embeddings = get_text_embeddings(categories, model, processor)
    
    return text_embeddings

def compute_category_similarities(cache_path='similarity_matrix.pt'):
    """Compute pairwise similarities between all Food101 categories using CLIP text embeddings"""
    # Check if cached similarity matrix exists
    if os.path.exists(cache_path):
        print(f"Loading cached similarity matrix from {cache_path}")
        cached_data = torch.load(cache_path)
        return (cached_data['similarity_matrix'].cuda(non_blocking=True))
    
    print("Computing similarity matrix...")
    # Get text embeddings for all categories
    text_embeddings = get_food101_text_embeddings()
    
    # Compute similarity matrix using dot product of normalized embeddings
    similarity_matrix = torch.mm(text_embeddings, text_embeddings.t())

    # Normalize to [0,1] range
    similarity_matrix = (similarity_matrix - torch.min(similarity_matrix)) / (torch.max(similarity_matrix) - torch.min(similarity_matrix))
    
    # Cache the results
    print(f"Saving similarity matrix to {cache_path}")
    torch.save({
        'similarity_matrix': similarity_matrix
    }, cache_path)
    
    return similarity_matrix.cuda(non_blocking=True)

if __name__ == "__main__":
    compute_category_similarities(cache_path='similarity_matrix.pt')  # Print first 5 values of first embedding