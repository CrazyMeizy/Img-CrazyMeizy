from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass, replace
from time import perf_counter

import matplotlib
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.datasets import FashionMNIST, MNIST

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MNIST_TASK = "MNIST"
FASHION_TASK = "FashionMNIST"

SEED = 71
BATCH_SIZE = 256
VALIDATION_SIZE = 0.1
MNIST_EPOCHS = 5
HEAD_EPOCHS = 5
FINETUNE_EPOCHS = 5
LEARNING_RATE = 0.001
FROZEN_BLOCKS = 2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
INITIAL_PATH = os.path.join(OUTPUT_DIR, "initial.pt")
BEST_PATH = os.path.join(OUTPUT_DIR, "best_fashion.pt")


@dataclass
class Dataset:
    train: DataLoader
    train_eval: DataLoader
    validation: DataLoader
    test: DataLoader
    class_names: list[str]


@dataclass
class Metrics:
    loss: float
    accuracy: float


@dataclass
class EpochResult:
    stage: str
    epoch: int
    task: str
    train_loss: float
    test_loss: float
    train_accuracy: float
    test_accuracy: float
    validation_accuracy: float


def load_dataset(dataset_type: type[MNIST]) -> Dataset:
    datasets = []
    for train in (True, False):
        source = dataset_type(DATA_DIR, train=train, download=True)
        images = source.data.unsqueeze(1).float().div_(255).sub_(0.5).div_(0.5)
        datasets.append(TensorDataset(images, source.targets))

    validation_size = int(len(datasets[0]) * VALIDATION_SIZE)
    train, validation = random_split(
        datasets[0], [len(datasets[0]) - validation_size, validation_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    return Dataset(
        train=DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator()),
        train_eval=DataLoader(train, batch_size=BATCH_SIZE),
        validation=DataLoader(validation, batch_size=BATCH_SIZE),
        test=DataLoader(datasets[1], batch_size=BATCH_SIZE),
        class_names=source.classes,
    )


class ConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(n_in, n_out, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
            for n_in, n_out in ((1, 16), (16, 32), (32, 64))
        ])
        self.heads = nn.ModuleDict()
        self.add_head(MNIST_TASK)

    def add_head(self, name: str) -> None:
        self.heads[name] = nn.Sequential(nn.Linear(64 * 3 * 3, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x: Tensor, task: str) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return self.heads[task](x.flatten(1))

    def set_trainable(self, task: str, frozen_blocks: int) -> None:
        for index, block in enumerate(self.blocks):
            block.requires_grad_(index >= frozen_blocks)
        for name, head in self.heads.items():
            head.requires_grad_(name == task)


class Trainer:
    def __init__(self, model: ConvNet, datasets: dict[str, Dataset], device: torch.device) -> None:
        self.model = model
        self.datasets = datasets
        self.device = device
        self.loss = nn.CrossEntropyLoss()
        self.history: list[EpochResult] = []
        self.best_accuracy = -1.0
        self.best_stage = ""
        self.best_epoch = 0

    def run_epoch(
            self, loader: DataLoader, task: str, optimizer: torch.optim.Optimizer | None = None,
    ) -> Metrics:
        is_training = optimizer is not None
        self.model.train(is_training)
        loss_sum, correct, count = 0.0, 0, 0
        with torch.set_grad_enabled(is_training):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x, task)
                loss = self.loss(logits, y)
                if is_training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                loss_sum += loss.item() * len(y)
                correct += (logits.argmax(1) == y).sum().item()
                count += len(y)
        return Metrics(loss_sum / count, correct / count)

    def fit(
            self, stage: str, task: str, epochs: int, frozen_blocks: int,
            optimizer: torch.optim.Optimizer, evaluate_tasks: tuple[str, ...],
            epoch_offset: int = 0,
    ) -> None:
        torch.manual_seed(SEED)
        self.model.set_trainable(task, frozen_blocks)
        print(f"\n{stage}: {task}, заморожено блоков: {frozen_blocks}", flush=True)
        for epoch in range(epochs + 1):
            task_epoch = epoch + epoch_offset
            if epoch > 0:
                self.datasets[task].train.generator.manual_seed(SEED + task_epoch)
                self.run_epoch(self.datasets[task].train, task, optimizer)
            for name in evaluate_tasks:
                data = self.datasets[name]
                train = self.run_epoch(data.train_eval, name)
                test = self.run_epoch(data.test, name)
                validation = self.run_epoch(data.validation, name)
                self.history.append(EpochResult(
                    stage, task_epoch, name, train.loss, test.loss,
                    train.accuracy, test.accuracy, validation.accuracy,
                ))
                if name == FASHION_TASK and epoch > 0 and validation.accuracy > self.best_accuracy:
                    self.best_accuracy = validation.accuracy
                    self.best_stage, self.best_epoch = stage, task_epoch
                    torch.save(self.model.state_dict(), BEST_PATH)
                print(f"  {task_epoch:2d}/{epochs + epoch_offset} {name:12s} CE={train.loss:.4f} "
                      f"train={train.accuracy:.4f} test={test.accuracy:.4f} "
                      f"val={validation.accuracy:.4f}", flush=True)


def save_csv(filename: str, rows: list[dict]) -> None:
    with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(history: list[EpochResult]) -> None:
    stages = list(dict.fromkeys(row.stage for row in history))
    combined = [
        replace(row, stage="06_07_transfer") if row.stage in ("06_frozen", "07_finetuned") else row
        for row in history
        if not (row.stage == "07_finetuned" and row.task == FASHION_TASK and row.epoch == HEAD_EPOCHS)
    ]
    groups = list(dict.fromkeys((row.stage, row.task) for row in history + combined))
    colors = dict(zip(groups, plt.get_cmap("tab10").colors))
    for selected in stages + ["06_07_transfer", "all"]:
        records = combined if selected in ("06_07_transfer", "all") else history
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        for stage, task in dict.fromkeys((row.stage, row.task) for row in records):
            if selected != "all" and stage != selected:
                continue
            rows = [row for row in records if row.stage == stage and row.task == task]
            for split, style in (("train", "-"), ("test", "--")):
                for ax, metric in zip(axes, ("loss", "accuracy")):
                    ax.plot(
                        [row.epoch for row in rows], [getattr(row, f"{split}_{metric}") for row in rows],
                        color=colors[stage, task], linestyle=style, label=f"{stage} / {task} / {split}",
                    )
        for ax, label in zip(axes, ("Перекрёстная энтропия", "Accuracy")):
            ax.set(xlabel="Эпоха обучения задачи", ylabel=label)
            ax.grid(alpha=0.25)
            if selected in ("06_07_transfer", "08_unfrozen", "all"):
                ax.axvline(HEAD_EPOCHS, color="gray", linestyle=":", label="Граница фаз переноса")
        axes[1].set_ylim(0, 1.02)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
        fig.suptitle("Все кривые обучения" if selected == "all" else selected)
        legend_rows = (len(labels) + 1) // 2
        fig.tight_layout(rect=(0, 0.04 * legend_rows + 0.03, 1, 0.95))
        fig.savefig(os.path.join(OUTPUT_DIR, f"{selected}_curves.png"), dpi=160)
        plt.close(fig)


@torch.no_grad()
def plot_similar_images(model: ConvNet, data: Dataset, device: torch.device) -> None:
    model.eval()
    probabilities = torch.cat([
        model(x.to(device), FASHION_TASK).softmax(1).cpu() for x, _ in data.test
    ])
    images, targets = data.test.dataset.tensors
    fig, axes = plt.subplots(10, 10, figsize=(16, 16))
    rows = []
    for c in range(10):
        candidates = torch.where(targets == c)[0]
        selected = candidates[probabilities[candidates].argmax(dim=0)]
        for t, index in enumerate(selected.tolist()):
            probability = probabilities[index, t].item()
            ax = axes[c, t]
            ax.imshow(images[index, 0], cmap="gray", vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{probability:.2g}", fontsize=9)
            if c == 0:
                ax.set_title(f"{data.class_names[t]}\n{probability:.2g}", fontsize=9)
            if t == 0:
                ax.set_ylabel(data.class_names[c], fontsize=9)
            rows.append({"class_c": c, "class_t": t, "test_index": index, "probability": probability,
                         "predicted_class": probabilities[index].argmax().item()})
    fig.suptitle("Строка: истинный класс c; столбец: целевой класс t; число: P(t | x)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(OUTPUT_DIR, "similar_images.png"), dpi=160)
    plt.close(fig)
    save_csv("similar_images.csv", rows)


def main() -> None:
    started = perf_counter()
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Устройство: {device}", flush=True)
    datasets: dict[str, Dataset] = {MNIST_TASK: load_dataset(MNIST), FASHION_TASK: load_dataset(FashionMNIST)}
    model = ConvNet().to(device)
    trainer = Trainer(model, datasets, device)
    both_tasks: tuple[str, ...] = (MNIST_TASK, FASHION_TASK)
    fashion_task: tuple[str, ...] = (FASHION_TASK,)

    # 3–4
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    trainer.fit("04_mnist", MNIST_TASK, MNIST_EPOCHS, 0, optimizer, (MNIST_TASK,))

    # 5
    model.add_head(FASHION_TASK)
    model.to(device)
    torch.save(model.state_dict(), INITIAL_PATH)

    # 6–7
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    trainer.fit("06_frozen", FASHION_TASK, HEAD_EPOCHS, 3, optimizer, fashion_task)
    trainer.fit("07_finetuned", FASHION_TASK, FINETUNE_EPOCHS, 0, optimizer, both_tasks,
                epoch_offset=HEAD_EPOCHS)

    # 8
    model.load_state_dict(torch.load(INITIAL_PATH, map_location=device, weights_only=True))
    transfer_epochs = HEAD_EPOCHS + FINETUNE_EPOCHS
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    trainer.fit("08_unfrozen", FASHION_TASK, transfer_epochs, 0, optimizer, both_tasks)

    # 9
    torch.manual_seed(SEED)
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            layer.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    trainer.fit("09_random", FASHION_TASK, transfer_epochs, FROZEN_BLOCKS, optimizer, fashion_task)

    # 10–11
    save_csv("history.csv", [asdict(row) for row in trainer.history])
    plot_curves(trainer.history)
    model.load_state_dict(torch.load(BEST_PATH, map_location=device, weights_only=True))
    test = trainer.run_epoch(datasets[FASHION_TASK].test, FASHION_TASK)
    save_csv("best_model.csv", [{
        "stage": trainer.best_stage, "epoch": trainer.best_epoch,
        "validation_accuracy": trainer.best_accuracy, "test_accuracy": test.accuracy, "test_loss": test.loss,
    }])
    plot_similar_images(model, datasets[FASHION_TASK], device)
    print(f"\nЛучшая модель: {trainer.best_stage}, эпоха {trainer.best_epoch}; "
          f"validation accuracy={trainer.best_accuracy:.4f}, test accuracy={test.accuracy:.4f}")
    print(f"Результаты: {OUTPUT_DIR}\nОбщее время: {perf_counter() - started:.1f} с")


if __name__ == "__main__":
    main()
