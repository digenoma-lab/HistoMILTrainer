"""Training functions"""
import torch
from torch import nn, optim
from histomil.utils import EarlyStopping
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, precision_recall_curve, classification_report,f1_score
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, val_loader, results_dir, learning_rate,
        bag_weight = 0.7, epochs = 20, patience = 2, stop_epoch = 2, class_weights = None):
    """
    Train function
    """
    if class_weights is not None:
        weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight = weights_tensor)
        print(criterion)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=patience, stop_epoch=stop_epoch, verbose=True)
    print("Start training")

    best_metrics = {}

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.
        train_preds = []
        train_labels = []
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            features, labels = features.to(device), labels.to(device)
            labels = labels.long()  # CrossEntropyLoss requires long type labels
            optimizer.zero_grad() #Initialize gradientes
            logits = model(features)
            loss = criterion(logits, labels) #Calculate loss
            total_loss = loss
            train_loss += total_loss.item()
            # Save metrics
            probs = torch.softmax(logits, dim=1)
            train_preds.extend(probs[:, 1].detach().cpu().numpy())
            train_labels.extend(labels.detach().cpu().numpy())

            loss.backward() #Backprop
            optimizer.step() #Optimizer step

        train_loss /= len(train_loader)

        train_auc = roc_auc_score(train_labels, train_preds)
        train_acc = accuracy_score(train_labels, np.array(train_preds) > 0.5)

        # Validate
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad(): #No grad
            for features, labels in tqdm(val_loader, desc="Validation"):
                features, labels = features.to(device), labels.to(device)
                features = features.squeeze(0) #Adds a batch dimension
                logits = model(features)
                #Reshape logits and labels for comparison
                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1).long()
                loss = criterion(logits, labels) #Calculate Cross entropy loss
                total_loss = loss
                val_loss += total_loss.item()
                probs = torch.softmax(logits, dim=1)
                val_preds.extend(probs[:, 1].cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_auc = roc_auc_score(val_labels, val_preds)
        val_acc = accuracy_score(val_labels, np.array(val_preds) > 0.5)

        print(f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f}.Train AUC: {train_auc:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}")

        early_stopping(epoch, val_loss, model, ckpt_name=f"{results_dir}/checkpoint.pt")

        # Save best epoch metrics
        if early_stopping.best_epoch == epoch:
            best_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_auc": train_auc,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_acc": val_acc,
            }

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    model.load_state_dict(torch.load(f"{results_dir}/checkpoint.pt"))
    return model, best_metrics

def test(model, test_loader):
    """Test function: Evaluates clam model with optimal threshold selection by F1 macro."""
    model.eval()
    all_labels, all_outputs = [], []
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc="Testing"):
            features, labels = features.to(device), labels.to(device)
            features = features.squeeze(0)  # CLAM asume batch size 1
            logits = model(features)
            logits = logits.view(-1, logits.size(-1)) #Reshape
            labels = labels.view(-1).long()
            probs = torch.softmax(logits, dim=1)  # logits: [1, 2]
            predicted = torch.argmax(probs, dim=1)  # predicted: [1]

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_outputs.append(probs[:, 1].cpu().numpy())  # prob. clase 1
            all_labels.append(labels.cpu().numpy())

    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)

    auc = roc_auc_score(all_labels, all_outputs)

    # Finding threshold
    best_f1_macro = 0
    best_threshold = 0.5
    thresholds = np.linspace(0, 1, 101)

    for thresh in thresholds:
        preds = (all_outputs >= thresh).astype(int)
        f1_macro = f1_score(all_labels, preds, average='macro')
        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
            best_threshold = thresh

    print(f"Optimal threshold for max F1 macro: {best_threshold:.3f} with F1 macro: {best_f1_macro:.3f}")

    pred_labels = (all_outputs >= best_threshold).astype(int)
    cm = confusion_matrix(all_labels, pred_labels)
    accuracy = accuracy_score(all_labels, pred_labels)

    print(f"Test AUC: {auc:.4f}")
    print(f"Test Accuracy (with optimal threshold): {accuracy:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(all_labels, pred_labels, digits=4))

    metrics = {
        "test_auc": auc,
        "test_acc": accuracy,
        "optimal_threshold": best_threshold,
        "f1_macro": best_f1_macro
    }
    return metrics
