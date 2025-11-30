from pathlib import Path
from textwrap import dedent
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(
    nbf.v4.new_markdown_cell(
        dedent(
            """
            # Face Recognition Transfer Learning (CNN + ViT)
            
            Notebook ini disiapkan khusus untuk Google Colab guna memenuhi requirement pada `Tugas Besar Deep Learning.pdf`. Pipeline mencakup:
            
            - Download + ekstraksi dataset wajah dari Google Drive (gdown).
            - Preprocessing dan face alignment menggunakan DeepFace agar rasio wajah konsisten.
            - Transfer learning dua arsitektur: EfficientNet (CNN) & Vision Transformer (ViT) lengkap dengan fine-tuning dan regularisasi agresif.
            - Evaluasi komprehensif (classification report, confusion matrix) serta demo inference.
            
            Jalankan setiap sel secara berurutan di runtime GPU (A100 / T4) Colab.
            """
        ).strip()
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "!nvidia-smi"
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """
            %%capture
            !pip install -q --upgrade pip
            !pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
            !pip install -q deepface timm albumentations==1.4.8 scikit-learn gdown seaborn opencv-python
            """
        ).strip()
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """
            import os
            import random
            import zipfile
            import shutil
            from pathlib import Path
            
            import numpy as np
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, Dataset
            from torchvision import datasets, transforms, models
            
            import timm
            from deepface import DeepFace
            import cv2
            import gdown
            from sklearn.model_selection import StratifiedShuffleSplit
            from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
            import seaborn as sns
            import matplotlib.pyplot as plt
            from PIL import Image
            from tqdm.auto import tqdm
            
            plt.style.use("seaborn-v0_8-colorblind")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("Device:", device)
            
            SEED = 1337
            IMG_SIZE = 224
            BATCH_SIZE = 32
            VAL_RATIO = 0.1
            TEST_RATIO = 0.1
            
            def seed_everything(seed: int = 42):
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = True
            
            seed_everything(SEED)
            
            DATA_DIR = Path("data_face_recognition")
            RAW_DIR = DATA_DIR / "dataset"
            PREPARED_DIR = DATA_DIR / "prepared_faces"
            CHECKPOINT_DIR = Path("checkpoints")
            for path in (DATA_DIR, RAW_DIR, PREPARED_DIR, CHECKPOINT_DIR):
                path.mkdir(parents=True, exist_ok=True)
            
            FILE_ID = "1tDo2zQC_1ZKY8aMYaalgr6nxRmBcRxaO"
            DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"
            ZIP_PATH = DATA_DIR / "dataset.zip"
            """
        ).strip()
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """
            def flatten_if_needed(root: Path):
                children = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.__')]
                if len(children) == 1 and (children[0] / 'Train').exists():
                    inner = children[0]
                    print(f"Menemukan folder tunggal {inner.name}, memindahkan isinya ke {root} ...")
                    for item in inner.iterdir():
                        shutil.move(str(item), root / item.name)
                    shutil.rmtree(inner)
            
            if not ZIP_PATH.exists():
                print("Mengunduh dataset...")
                gdown.download(DOWNLOAD_URL, str(ZIP_PATH), quiet=False)
            else:
                print("File ZIP sudah ada, skip download.")
            
            if not any(RAW_DIR.iterdir()):
                print("Mengekstrak dataset...")
                with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
                    zf.extractall(RAW_DIR)
            else:
                print("Dataset sudah diekstrak.")
            
            flatten_if_needed(RAW_DIR)
            for sub in RAW_DIR.iterdir():
                if sub.is_dir():
                    flatten_if_needed(sub)
            
            macos_dirs = list(RAW_DIR.rglob('_MacOs'))
            train_dir = RAW_DIR / 'Train'
            if macos_dirs and train_dir.exists():
                for mac_dir in macos_dirs:
                    print(f"Menggabungkan isi {mac_dir} ke {train_dir} ...")
                    for class_dir in mac_dir.iterdir():
                        if not class_dir.is_dir():
                            continue
                        target_dir = train_dir / class_dir.name
                        target_dir.mkdir(parents=True, exist_ok=True)
                        for item in class_dir.glob('*'):
                            if item.is_dir():
                                continue
                            dest = target_dir / item.name
                            if dest.exists():
                                dest = target_dir / f"{item.stem}_mac{item.suffix}"
                            shutil.move(str(item), dest)
                    shutil.rmtree(mac_dir)
                print("Folder _MacOs selesai digabungkan.")
            else:
                print("Tidak menemukan folder _MacOs atau Train belum ada.")
            
            print("Isi folder dataset (level 1):")
            for path in RAW_DIR.iterdir():
                print(' -', path)
            """
        ).strip()
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """
            VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            
            def detect_split_dirs(source_root: Path):
                candidates = [d for d in source_root.iterdir() if d.is_dir() and not d.name.startswith('.__')]
                split_dirs = []
                for cand in candidates:
                    if any(child.is_dir() for child in cand.iterdir()):
                        split_dirs.append(cand)
                if not split_dirs:
                    split_dirs = [source_root]
                return split_dirs
            
            def align_and_prepare(source_root: Path, target_root: Path, detector: str = 'retinaface', force: bool = False):
                sample_exists = target_root.exists() and any(target_root.rglob('*.jpg'))
                if sample_exists and not force:
                    print('Dataset ter-align sudah tersedia, skip langkah ini.')
                    return
                target_root.mkdir(parents=True, exist_ok=True)
                for split_dir in detect_split_dirs(source_root):
                    split_name = 'Train' if split_dir == source_root else split_dir.name
                    class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
                    for class_dir in class_dirs:
                        image_paths = [p for p in class_dir.rglob('*') if p.suffix.lower() in VALID_EXTS]
                        if not image_paths:
                            continue
                        dest_dir = target_root / split_name / class_dir.name
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        for img_path in tqdm(image_paths, desc=f"{split_name}-{class_dir.name}", leave=False):
                            dest_path = dest_dir / f"{img_path.stem}.jpg"
                            if dest_path.exists():
                                continue
                            try:
                                faces = DeepFace.extract_faces(img_path=str(img_path), detector_backend=detector, enforce_detection=False)
                                if not faces:
                                    shutil.copy(str(img_path), dest_path)
                                    continue
                                face = faces[0]['face']
                                if face.max() <= 1.0:
                                    face = (face * 255).astype('uint8')
                                face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                                cv2.imwrite(str(dest_path), face_bgr)
                            except Exception as exc:
                                print(f"[WARNING] {img_path.name}: {exc}")
                print('Face alignment selesai, dataset siap digunakan.')
            
            align_and_prepare(RAW_DIR, PREPARED_DIR)
            """
        ).strip()
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """
            class FaceSubset(Dataset):
                def __init__(self, base_dataset: datasets.ImageFolder, indices, transform):
                    self.base_dataset = base_dataset
                    self.indices = np.array(indices)
                    self.transform = transform
                    self.samples = [base_dataset.samples[i] for i in self.indices]
                    self.targets = [base_dataset.targets[i] for i in self.indices]
                
                def __len__(self):
                    return len(self.indices)
                
                def __getitem__(self, idx):
                    path, label = self.samples[idx]
                    image = self.base_dataset.loader(path)
                    if self.transform:
                        image = self.transform(image)
                    return image, label
            
            train_root = PREPARED_DIR / 'Train'
            if not train_root.exists():
                raise FileNotFoundError('Folder Train tidak ditemukan setelah preprocessing. Pastikan struktur dataset benar.')
            
            train_transform = transforms.Compose([
                transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.15, 0.15, 0.15, 0.05)], p=0.3),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
                transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            eval_transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            base_dataset = datasets.ImageFolder(str(train_root))
            class_names = base_dataset.classes
            num_classes = len(class_names)
            print(f"Total kelas: {num_classes} -> {class_names}")
            
            indices = np.arange(len(base_dataset))
            targets = np.array(base_dataset.targets)
            sss_primary = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO + TEST_RATIO, random_state=SEED)
            train_idx, valtest_idx = next(sss_primary.split(indices, targets))
            relative_test_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
            sss_secondary = StratifiedShuffleSplit(n_splits=1, test_size=relative_test_ratio, random_state=SEED)
            val_rel_idx, test_rel_idx = next(sss_secondary.split(valtest_idx, targets[valtest_idx]))
            val_idx = valtest_idx[val_rel_idx]
            test_idx = valtest_idx[test_rel_idx]
            
            train_subset = FaceSubset(base_dataset, train_idx, train_transform)
            val_subset = FaceSubset(base_dataset, val_idx, eval_transform)
            internal_test_subset = FaceSubset(base_dataset, test_idx, eval_transform)
            
            external_test_root = PREPARED_DIR / 'Test'
            if external_test_root.exists() and any(external_test_root.iterdir()):
                print('Menggunakan folder Test eksternal untuk evaluasi.')
                test_dataset = datasets.ImageFolder(str(external_test_root), transform=eval_transform)
                test_samples = [s[0] for s in test_dataset.samples]
            else:
                print('Folder Test tidak tersedia, memakai split internal.')
                test_dataset = internal_test_subset
                test_samples = [s[0] for s in internal_test_subset.samples]
            
            train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
            
            train_counts = np.bincount(train_subset.targets, minlength=num_classes)
            class_weights = len(train_subset.targets) / (num_classes * torch.tensor(train_counts, dtype=torch.float32))
            CLASS_WEIGHTS = class_weights
            print('Distribusi kelas train:', train_counts)
            print('Class weights:', class_weights.numpy())
            """
        ).strip()
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """
            def evaluate_on_loader(model, data_loader, criterion):
                model.eval()
                total = 0
                loss_sum = 0.0
                correct = 0
                with torch.no_grad():
                    for inputs, labels in data_loader:
                        inputs = inputs.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss_sum += loss.item() * inputs.size(0)
                        preds = outputs.argmax(dim=1)
                        correct += (preds == labels).sum().item()
                        total += inputs.size(0)
                return loss_sum / total, correct / total
            
            def plot_history(history, title):
                epochs = [h['epoch'] for h in history]
                train_loss = [h['train_loss'] for h in history]
                val_loss = [h['val_loss'] for h in history]
                train_acc = [h['train_acc'] for h in history]
                val_acc = [h['val_acc'] for h in history]
                plt.figure(figsize=(12,4))
                plt.subplot(1,2,1)
                plt.plot(epochs, train_loss, label='Train')
                plt.plot(epochs, val_loss, label='Val')
                plt.title(f'{title} Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.subplot(1,2,2)
                plt.plot(epochs, train_acc, label='Train')
                plt.plot(epochs, val_acc, label='Val')
                plt.title(f'{title} Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.show()
            
            def build_cnn_model(num_classes: int) -> nn.Module:
                model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
                for param in model.features.parameters():
                    param.requires_grad = False
                in_features = model.classifier[1].in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(0.45),
                    nn.Linear(in_features, 512),
                    nn.GELU(),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.25),
                    nn.Linear(512, num_classes)
                )
                return model
            
            def build_vit_model(num_classes: int) -> nn.Module:
                model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
                for name, param in model.named_parameters():
                    if not name.startswith('head'):
                        param.requires_grad = False
                in_features = model.head.in_features
                model.head = nn.Sequential(
                    nn.LayerNorm(in_features),
                    nn.Linear(in_features, num_classes)
                )
                return model
            
            def train_model(model, train_loader, val_loader, *, epochs=15, lr=3e-4, weight_decay=1e-4, unfreeze_at=3, model_name='model'):
                model = model.to(device)
                history = []
                best_acc = 0.0
                best_path = CHECKPOINT_DIR / f"{model_name}_best.pth"
                criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device), label_smoothing=0.1)
                optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
                for epoch in range(1, epochs + 1):
                    if unfreeze_at and epoch == unfreeze_at:
                        for param in model.parameters():
                            param.requires_grad = True
                        optimizer = torch.optim.AdamW(model.parameters(), lr=lr * 0.25, weight_decay=weight_decay / 2)
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - epoch + 1)
                        print(f"[{model_name}] Backbone di-unfreeze pada epoch {epoch}.")
                    model.train()
                    train_loss = 0.0
                    train_correct = 0
                    total = 0
                    for inputs, labels in tqdm(train_loader, desc=f"{model_name} Epoch {epoch}/{epochs}", leave=False):
                        inputs = inputs.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                        scaler.scale(loss).backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        scaler.step(optimizer)
                        scaler.update()
                        train_loss += loss.item() * inputs.size(0)
                        preds = outputs.argmax(dim=1)
                        train_correct += (preds == labels).sum().item()
                        total += inputs.size(0)
                    scheduler.step()
                    train_loss /= total
                    train_acc = train_correct / total
                    val_loss, val_acc = evaluate_on_loader(model, val_loader, criterion)
                    history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})
                    if val_acc > best_acc:
                        best_acc = val_acc
                        torch.save({'model_state': model.state_dict(), 'val_acc': val_acc, 'epoch': epoch}, best_path)
                        print(f"[{model_name}] 🔥 val_acc meningkat ke {val_acc:.4f} (epoch {epoch}).")
                    else:
                        print(f"[{model_name}] val_acc {val_acc:.4f} | train_acc {train_acc:.4f}")
                return history, best_path
            
            def evaluate_checkpoint(build_fn, checkpoint_path, data_loader, class_names, title='Model'):
                model = build_fn(len(class_names)).to(device)
                state = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(state['model_state'])
                model.eval()
                preds, labels_list = [], []
                with torch.no_grad():
                    for inputs, labels in tqdm(data_loader, desc=f"Eval {title}", leave=False):
                        inputs = inputs.to(device, non_blocking=True)
                        outputs = model(inputs)
                        preds.extend(outputs.argmax(dim=1).cpu().numpy())
                        labels_list.extend(labels.cpu().numpy())
                acc = accuracy_score(labels_list, preds)
                print(f"{title} accuracy: {acc:.4f}")
                print(classification_report(labels_list, preds, target_names=class_names))
                cm = confusion_matrix(labels_list, preds)
                plt.figure(figsize=(6,5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
                plt.title(f"{title} Confusion Matrix")
                plt.ylabel('True')
                plt.xlabel('Predicted')
                plt.show()
                return acc
            """
        ).strip()
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """
            cnn_model = build_cnn_model(num_classes)
            cnn_history, cnn_ckpt = train_model(
                cnn_model,
                train_loader,
                val_loader,
                epochs=18,
                lr=3e-4,
                weight_decay=1e-4,
                unfreeze_at=4,
                model_name='cnn_efficientnet'
            )
            plot_history(cnn_history, 'EfficientNet-B4 (CNN)')
            """
        ).strip()
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """
            vit_model = build_vit_model(num_classes)
            vit_history, vit_ckpt = train_model(
                vit_model,
                train_loader,
                val_loader,
                epochs=22,
                lr=2e-4,
                weight_decay=5e-5,
                unfreeze_at=6,
                model_name='vit_base'
            )
            plot_history(vit_history, 'ViT-B/16')
            """
        ).strip()
    )
)
cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """
            print('Evaluasi EfficientNet-B4 pada set test:')
            cnn_test_acc = evaluate_checkpoint(build_cnn_model, cnn_ckpt, test_loader, class_names, title='CNN EfficientNet-B4')
            print('\nEvaluasi ViT-B/16 pada set test:')
            vit_test_acc = evaluate_checkpoint(build_vit_model, vit_ckpt, test_loader, class_names, title='ViT-B/16')
            print(f"Ringkasan -> CNN: {cnn_test_acc:.4f} | ViT: {vit_test_acc:.4f}")
            """
        ).strip()
    )
)

cells.append(
    nbf.v4.new_code_cell(
        dedent(
            """
            def predict_image(model, image_path: str, transform):
                image = Image.open(image_path).convert('RGB')
                tensor = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred_idx = probs.argmax()
                return pred_idx, probs[pred_idx], probs
            
            sample_paths = random.sample(test_samples, k=min(3, len(test_samples)))
            print('Menampilkan prediksi untuk sampel:', sample_paths)
            
            cnn_infer = build_cnn_model(num_classes).to(device)
            cnn_infer.load_state_dict(torch.load(cnn_ckpt, map_location=device)['model_state'])
            cnn_infer.eval()
            vit_infer = build_vit_model(num_classes).to(device)
            vit_infer.load_state_dict(torch.load(vit_ckpt, map_location=device)['model_state'])
            vit_infer.eval()
            
            for path in sample_paths:
                label_name = Path(path).parent.name
                cnn_idx, cnn_conf, cnn_probs = predict_image(cnn_infer, path, eval_transform)
                vit_idx, vit_conf, vit_probs = predict_image(vit_infer, path, eval_transform)
                print(f"\nFile: {path}")
                print(f"Label GT : {label_name}")
                print(f"CNN Pred : {class_names[cnn_idx]} ({cnn_conf:.3f})")
                print(f"ViT Pred : {class_names[vit_idx]} ({vit_conf:.3f})")
            """
        ).strip()
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        dedent(
            """
            ## Catatan Pengembangan Lanjutan
            
            - Silakan aktifkan `force=True` pada fungsi `align_and_prepare` bila ingin re-build dataset setelah menambah gambar baru.
            - Tambahkan augmentasi spesifik (CutMix/MixUp) pada loader untuk menangani kelas dengan data sedikit.
            - Notebook ini menyimpan checkpoint terbaik di folder `checkpoints/`. Unggah ke Drive bila perlu inference di sesi berbeda.
            - Untuk deployment, export model menjadi TorchScript / ONNX atau buat pipeline embedding + ANN index menggunakan bobot terbaik.
            """
        ).strip()
    )
)

nb['cells'] = cells
nb['metadata'] = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {
        'name': 'python',
        'version': '3.10'
    }
}

output_path = Path('face_recognition_transfer_learning.ipynb')
with output_path.open('w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'Notebook ditulis ke {output_path.resolve()}')
