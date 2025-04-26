from pathlib import Path
import torch


def save_checkpoint(epoch, model, optimizer, best_loss, output_dir: Path, checkpoint_name):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss
    }
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / f'{checkpoint_name}.pth'
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved at epoch {epoch}')


def load_checkpoint(model, device, optimizer, output_dir: Path, checkpoint_name):
    checkpoint_path = output_dir / f'checkpoints/{checkpoint_name}.pth'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        print(f'Checkpoint loaded from epoch {checkpoint["epoch"]}')
        return start_epoch, best_loss, model
    return 1, float('inf'), model


def save_weights(model, save_name, output_dir: Path):
    output_dir = output_dir / 'weights'
    output_dir.mkdir(exist_ok=True, parents=True)
    weights_path = output_dir / f'{save_name}.pth'
    torch.save(model.state_dict(), weights_path)
    print(f'Model saved: {save_name}')

