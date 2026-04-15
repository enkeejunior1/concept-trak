import os
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import json

from utils import (
    SyntheticClassDataset,
    create_model,
    check_gpu_health_and_set_device,
    load_pipeline,
    flush,
    seed_everything,
    MultiLabelResNet,
    num_conds,
    num_class,
)

# Import classifier from 0train_classifier.py
import torch.nn as nn
import torchvision.models as models
concept_list = ["shape", "color"]

# Set default dtype
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('medium')

@torch.no_grad()
def evaluate_images_with_classifier(images, classifier, transform, device, true_labels):
    """Evaluate generated images with trained classifier"""
    classifier.eval()
    
    # Classify all images
    image_tensors = []
    for image in images:
        img_tensor = transform(image)
        image_tensors.append(img_tensor)
    image_batch = torch.stack(image_tensors, dim=0).to(device)
    logits = classifier(image_batch)
    preds = [torch.argmax(logit, dim=1) for logit in logits]
    
    # Calculate accuracies
    all_predictions = []
    all_accuracies = []
    
    batch_size = len(images)
    for i in range(batch_size):
        # Get predictions for this image
        img_preds = [pred[i].item() for pred in preds]
        all_predictions.append(img_preds)
        
        # Calculate accuracies for this image
        accs = []
        for j in range(len(img_preds)):
            acc = 1.0 if img_preds[j] == true_labels[j] else 0.0
            accs.append(acc)
        overall_acc = sum(accs) / len(accs)
        
        accuracies = {
            f'{concept_list[j]}_acc': accs[j] for j in range(len(concept_list))
        }
        accuracies['overall_acc'] = overall_acc
        all_accuracies.append(accuracies)
    return all_predictions, all_accuracies

def generate_images_until_perfect(pipe, cond_tensor, cond_tensor_neg, generator, args, 
                                 classifier, classifier_transform, device, true_labels, max_attempts=10):
    """Generate images until all have 100% accuracy or max attempts reached"""
    attempt = 0
    best_images = None
    best_predictions = None
    best_accuracies = None
    best_overall_score = 0.0
    best_attempt = 0
    best_init_noise = None
    
    while attempt < max_attempts:
        attempt += 1
        print(f"  Attempt {attempt}/{max_attempts}")
        
        # Generate images and store init noise
        with torch.no_grad():
            images, init_noise = pipe(
                batch_size=args.batch_size, 
                num_inference_steps=args.num_inference_steps,
                conditions=cond_tensor.float(), 
                null_cond=cond_tensor_neg.float(), 
                generator=generator.manual_seed(42 + attempt), 
                guidance_scale=args.guidance_scale,
                return_init_noise=True,
            )

        # Evaluate images with classifier
        predictions, accuracies = evaluate_images_with_classifier(
            images, classifier, classifier_transform, device, true_labels
        )
        
        # Calculate overall score for this attempt
        overall_score = sum(acc['overall_acc'] for acc in accuracies) / len(accuracies)
        perfect_count = sum(1 for acc in accuracies if acc['overall_acc'] == 1.0)
        
        # Check if this is the best attempt so far
        if overall_score > best_overall_score:
            best_images = images
            best_predictions = predictions
            best_accuracies = accuracies
            best_overall_score = overall_score
            best_attempt = attempt
            best_init_noise = init_noise
            print(f"  New best: {perfect_count}/{len(accuracies)} perfect images, overall score: {overall_score:.3f}")
        else:
            print(f"  {perfect_count}/{len(accuracies)} perfect images, overall score: {overall_score:.3f}")
        
        # Check if all images have 100% accuracy
        all_perfect = all(acc['overall_acc'] == 1.0 for acc in accuracies)
        
        if all_perfect:
            print(f"  All images achieved 100% accuracy on attempt {attempt}")
            return images, predictions, accuracies, attempt, init_noise
        
        flush()  # Clear GPU memory before next attempt
    
    print(f"  Max attempts ({max_attempts}) reached. Using best results from attempt {best_attempt} (score: {best_overall_score:.3f}).")
    return best_images, best_predictions, best_accuracies, best_attempt, best_init_noise

def main():
    """Main function for generating images from trained model."""
    experiment_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default=str(experiment_dir), help='toy experiment directory')
    parser.add_argument('--model_path', type=str, default=str(experiment_dir / 'weights' / 'model.bin'), help='path to trained model')
    parser.add_argument('--ema_path', type=str, required=False, help='path to ema model')
    parser.add_argument('--classifier_path', type=str, default=str(experiment_dir / 'weights' / 'classifier.bin'), help='path to trained classifier')
    parser.add_argument('--output_dir', type=str, default=str(experiment_dir / 'results' / 'generated_samples'), help='output directory for generated images')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for generation')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=2.0, help='guidance scale for classifier-free guidance')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--max_attempts', type=int, default=10, help='maximum attempts to achieve 100% accuracy per condition')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create noise directory
    noise_dir = os.path.join(args.output_dir, 'init_noise')
    os.makedirs(noise_dir, exist_ok=True)
    
    # Set device
    device = check_gpu_health_and_set_device(0)
    
    # Set seed
    seed_everything(args.seed)

    # Load dataset info (for conditioning dimensions)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Classifier transform (without normalization to match training)
    classifier_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    imgs_path = os.path.join(args.base_dir, "data/images")
    attr_path = os.path.join(args.base_dir, "data/labels/metadata.npy")
    ds = SyntheticClassDataset(imgs_path=imgs_path, attr_path=attr_path, num_conds=num_conds, num_class=num_class, transform=transform)

    # Create and load classifier
    from utils import MultiLabelResNet
    classifier = MultiLabelResNet(num_classes_per_attribute=num_class, num_attributes=num_conds).to(device)
    classifier.load_state_dict(torch.load(args.classifier_path, map_location=device))
    classifier.eval()
    print(f"Classifier loaded from {args.classifier_path}")

    # Create model
    model = create_model(device, cond_dim=ds.num_conds*ds.num_class)
    
    # Load trained weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    print(f"Model loaded from {args.model_path}")

    # Load the Diffusion pipeline
    pipe = load_pipeline(model, device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Generate conditions based on concept_list (shape | color)
    conds = [
        [0, 0],  # OOD
        [0, 1],  # ID
        [1, 0],  # ID
        [1, 1],  # ID
    ]

    # Convert conditions to tensor format
    cond_tensor = torch.tensor(conds, device=device)
    cond_tensor = torch.nn.functional.one_hot(
        cond_tensor.long(), num_classes=ds.num_class
    ).float().flatten(start_dim=1)
    cond_tensor_list = cond_tensor.split(1, dim=0)
    print(f"Generating {len(conds)} images...")

    # Store all metadata
    all_metadata = []
    total_attempts = []

    for cond_idx, (cond_list, cond_tensor) in enumerate(zip(conds, cond_tensor_list)):
        cond_label = "_".join([str(c) for c in cond_list])
        print(f"Generating condition {cond_idx+1}/{len(conds)}: {cond_label}")
        cond_tensor = cond_tensor.repeat(args.batch_size, 1)
        cond_tensor_neg = torch.zeros_like(cond_tensor, dtype=torch.float32).flatten(start_dim=1)

        # Generate images until perfect accuracy
        images, predictions, accuracies, attempts, init_noise = generate_images_until_perfect(
            pipe, cond_tensor, cond_tensor_neg, generator, args,
            classifier, classifier_transform, device, cond_list, args.max_attempts
        )
        
        total_attempts.append(attempts)
        
        # Save generated images, init noise and metadata
        for batch_idx, (image, pred, acc, noise) in enumerate(zip(images, predictions, accuracies, init_noise)):
            filename = f'{batch_idx}-{cond_label}.png'
            save_path = os.path.join(args.output_dir, filename)
            image.save(save_path)
            
            # Save init noise
            noise_filename = f'{batch_idx}-{cond_label}.pt'
            noise_path = os.path.join(noise_dir, noise_filename)
            torch.save(noise.cpu(), noise_path)
            
            # Store metadata
            metadata = {
                'filename': filename,
                'noise_filename': noise_filename,
                'true_labels': cond_list,
                'predicted_labels': pred,
                'accuracies': acc,
                'condition_label': cond_label,
                'attempts_needed': attempts
            }
            all_metadata.append(metadata)
            
            print(f"Saved: {save_path}, Noise: {noise_path}, Acc: {acc['overall_acc']:.3f}")

    # Save metadata to JSON file
    metadata_path = os.path.join(args.output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    # Calculate and save summary statistics based on concept_list
    concept_accs = {}
    for i, concept in enumerate(concept_list):
        concept_accs[f'{concept}_accuracy'] = sum(m['accuracies'][f'{concept}_acc'] for m in all_metadata) / len(all_metadata)
    
    total_overall_acc = sum(m['accuracies']['overall_acc'] for m in all_metadata) / len(all_metadata)
    
    perfect_images = sum(1 for m in all_metadata if m['accuracies']['overall_acc'] == 1.0)
    avg_attempts = sum(total_attempts) / len(total_attempts)
    
    summary = {
        'total_images': len(all_metadata),
        'perfect_accuracy_images': perfect_images,
        'perfect_accuracy_rate': perfect_images / len(all_metadata),
        'average_attempts_per_condition': avg_attempts,
        'average_accuracies': {
            **concept_accs,
            'overall_accuracy': total_overall_acc
        }
    }
    
    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"All images saved to {args.output_dir}")
    print(f"Init noise saved to {noise_dir}")
    print(f"Metadata saved to {metadata_path}")
    print(f"Summary saved to {summary_path}")
    
    # Print accuracy for each concept dynamically
    concept_acc_str = ", ".join([f"{concept}: {concept_accs[f'{concept}_accuracy']:.3f}" for concept in concept_list])
    print(f"Overall Accuracy: {total_overall_acc:.3f} ({concept_acc_str})")
    print(f"Perfect Accuracy Rate: {perfect_images}/{len(all_metadata)} ({perfect_images/len(all_metadata)*100:.1f}%)")
    print(f"Average Attempts per Condition: {avg_attempts:.1f}")

if __name__ == "__main__":
    main()