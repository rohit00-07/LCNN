# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix

from dataset import (
    create_splits,
    ASVSpoofDataset,
    asvspoof_collate_fn,
)
from model import LCNN


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct = 0
    total = 0

    for feats, labels in dataloader:
        feats = feats.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(feats)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * feats.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    for feats, labels in dataloader:
        feats = feats.to(device)
        labels = labels.to(device)

        outputs = model(feats)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * feats.size(0)

        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    avg_loss = running_loss / total
    acc = correct / total

    # --- NEW METRICS ---
    from sklearn.metrics import precision_score, recall_score, f1_score

    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)

    return avg_loss, acc, all_labels, all_preds, precision, recall, f1

def main():
    # ---------------- CONFIG ----------------
    data_root = "data"
    splits_dir = "splits"
    sample_rate = 16000
    n_lfcc = 60
    batch_size = 16
    num_epochs = 30
    lr = 1e-3
    num_classes = 2
    seed = 42
    # ----------------------------------------

    torch.manual_seed(seed)

    # Create splits if not present
    if not (
        os.path.exists(os.path.join(splits_dir, "train.txt"))
        and os.path.exists(os.path.join(splits_dir, "val.txt"))
        and os.path.exists(os.path.join(splits_dir, "test.txt"))
    ):
        print("Creating train/val/test splits...")
        create_splits(data_root=data_root, splits_dir=splits_dir, seed=seed)

    # Datasets
    train_dataset = ASVSpoofDataset(
        split_file=os.path.join(splits_dir, "train.txt"),
        data_root=data_root,
        sample_rate=sample_rate,
        n_lfcc=n_lfcc,
    )
    val_dataset = ASVSpoofDataset(
        split_file=os.path.join(splits_dir, "val.txt"),
        data_root=data_root,
        sample_rate=sample_rate,
        n_lfcc=n_lfcc,
    )
    test_dataset = ASVSpoofDataset(
        split_file=os.path.join(splits_dir, "test.txt"),
        data_root=data_root,
        sample_rate=sample_rate,
        n_lfcc=n_lfcc,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=asvspoof_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=asvspoof_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=asvspoof_collate_fn,
    )

    # Device & model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LCNN(in_channels=1, num_classes=num_classes).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # -------------- TRAINING LOOP --------------
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, _, _, _, _, _ = evaluate(model, val_loader, device)

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        # Save best model (by val acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_lfcc_lcnn.pt")

    print(f"Training done. Best Val Acc: {best_val_acc:.4f}")

    # -------------- FINAL EVALUATION WITH METRICS --------------
    model.load_state_dict(torch.load("best_lfcc_lcnn.pt", map_location=device))

    # ---- Validation Metrics ----
    val_loss, val_acc, val_labels, val_preds, val_prec, val_rec, val_f1 = evaluate(model, val_loader, device)
    cm_val = confusion_matrix(val_labels, val_preds, labels=[0, 1])

    print("\nValidation Metrics:")
    print(f"Accuracy:  {val_acc:.4f}")
    print(f"Precision: {val_prec:.4f}")
    print(f"Recall:    {val_rec:.4f}")
    print(f"F1-score:  {val_f1:.4f}")

    print("\nConfusion Matrix (Validation):")
    print(cm_val)
    print("Rows = true [0=bona_fide, 1=spoof], Cols = predicted\n")

    # ---- Test Metrics ----
    test_loss, test_acc, test_labels, test_preds, test_prec, test_rec, test_f1 = evaluate(model, test_loader, device)
    cm_test = confusion_matrix(test_labels, test_preds, labels=[0, 1])

    print("Test Metrics:")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall:    {test_rec:.4f}")
    print(f"F1-score:  {test_f1:.4f}")

    print("\nConfusion Matrix (Test):")
    print(cm_test)
    print("Rows = true [0=bona_fide, 1=spoof], Cols = predicted")


if __name__ == "__main__":
    main()