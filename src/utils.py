import time
from typing import Callable, Any
import typing
from pytorch_lightning import LightningModule, LightningDataModule

import torch
from torch.utils.data import DataLoader
from collections import Counter
from data.fashionmnist_loader import (
    FashionMNISTDataModule,
    ConceptFashionMNISTDataModule,
)
from data.celeba_loader import CelebADataModule
from data.CUB_loader import CUBDataModule
from src.models import (
    FashionMNIST_for_CBM,
    Template_CBM_MultiClass,
    C2Y_model,
    UtoY_model,
    Standard_resnet18,
)
from typing import Type

import yaml  # type: ignore
from pathlib import Path
import csv
import os

import functools


@typing.no_type_check
def total_time_function(func: Callable):
    """Decorator to measure time of a function across multiple calls


    Example usage

    @time_function
    def my_function():
        time.sleep(0.5)  # Simulate work


    # Simulate calling the function multiple times
    for _ in range(10):
        my_function()

    # Now, you can access the total execution time from the wrapper
    print(f"Total execution time: {my_function.total_time} seconds")

    """
    total_time = 0

    def wrapper(*args, **kwargs):
        nonlocal total_time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        call_duration = end_time - start_time
        total_time += call_duration

        # Print time taken for this call and the accumulated total time
        # print(f"Time taken for this call: {call_duration:.4f} seconds")

        # Store total time in the wrapper but don't print immediately
        wrapper.total_time = total_time
        return result

    wrapper.total_time = 0  # Initialize total time
    wrapper.print_total_time = lambda: print(
        f"\nTotal execution time: {wrapper.total_time:.4f} seconds\n"
    )  # Print total time when needed
    return wrapper


class TimeoutError(Exception):
    pass


def timeout(seconds=300, error_message="Function call timed out"):
    import threading

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(seconds)
            if thread.is_alive():
                thread.join(0)  # Ensure thread cleanup
                raise TimeoutError(error_message)
            if exception[0]:
                raise exception[0]
            return result[0]

        return wrapper

    return decorator


# Load configuration from YAML file
def load_config(config_file: Path) -> dict[str, Any]:
    with open(config_file, "r") as file:
        return yaml.safe_load(file)


@typing.no_type_check
def analyze_dataset(dataloader: DataLoader) -> None:
    # Initialize a Counter to count labels
    label_counts = Counter()
    concept_counts = torch.Tensor()
    total_samples = 0

    # Iterate over the dataloader
    for _, concept_vector, labels in dataloader:
        label_counts.update(labels.tolist())
        total_samples += len(labels)

        if len(concept_counts) == 0:
            concept_counts = torch.zeros(concept_vector.size(1))

        # Sum up the occurrences of '1' for each component in y1 across batches
        concept_counts += concept_vector.sum(dim=0)

    # Calculate and print proportions
    print("Label Proportions:")
    for label, count in label_counts.items():
        proportion = count / total_samples
        print(f"Label {label}: {proportion:.2%}")

    # Calculate and print proportions for each component of y1
    print("\n Percentage of concept value being 1:")
    for idx, count in enumerate(concept_counts):
        proportion = count.item() / total_samples
        print(f"Concept {idx}: {proportion:.2%}")


def get_component_with_dicts(
    component_type: str, name: str
) -> Type[LightningModule] | Type[LightningDataModule]:
    """
    Returns the appropriate component (model or dataset) based on the component_type and name.

    Args:
        component_type (str): The type of the component ('model' or 'dataset').
        name (str): The name of the component to use.

    Returns:
        torch.nn.Module: The requested model or dataset.

    Raises:
        ValueError: If the provided name is not supported for the specified component type.
    """

    # Define model and dataset registries
    model_dict = {
        "Standard_FashionMNIST": FashionMNIST_for_CBM,
        "Standard_CelebA": Standard_resnet18,
        "Standard_CUB": Standard_resnet18,
        "UtoY_model": UtoY_model,
        "CBM": Template_CBM_MultiClass,
        "c2y": C2Y_model,
        # Add more models as needed
    }

    dataset_dict = {
        "FMNIST": FashionMNISTDataModule,
        "Concept_FMNIST": ConceptFashionMNISTDataModule,
        "Complete_Concept_FMNIST": ConceptFashionMNISTDataModule,
        "CelebA": CelebADataModule,
        "CUB": CUBDataModule,
    }

    if component_type == "model":
        component_dict = model_dict
    elif component_type == "dataset":
        component_dict = dataset_dict

    else:
        raise ValueError("component_type must be 'model' or 'dataset'.")

    if name not in component_dict:
        raise ValueError(
            f"{component_type} '{name}' not supported. Choose from {list(component_dict.keys())}"
        )

    component_class = component_dict[name]

    return component_class


def dict_to_csv(
    metric_dict: dict, dir_path: Path, config_path: Path, parents: bool = False
) -> None:
    file_name = os.path.splitext(os.path.basename(config_path))[0]
    # Create output directory if it doesn't exist
    output_dir = Path(dir_path)

    try:
        output_dir.mkdir(exist_ok=True, parents=parents)
    except FileNotFoundError:
        os.makedirs(output_dir, exist_ok=True)
    print("Final Results found at:", (dir_path / (file_name + ".csv")))
    with open((dir_path / file_name).with_suffix(".csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write headers (keys)
        writer.writerow(metric_dict.keys())
        # Write values as a single row
        writer.writerow(metric_dict.values())


def count_maskedmlp_params(model):
    total_active = 0
    total_masked = 0
    from zuko.nn import MaskedLinear

    for i, module in enumerate(model.modules()):
        if isinstance(module, MaskedLinear):
            mask = module.mask
            active = int(mask.sum().item())  # ones in mask
            masked = mask.numel() - active  # zeros in mask

            total_active += active
            total_masked += masked

    return total_masked


def run_benchmark(model, data_loader):
    # ensure everyone has 1 worker
    data_loader.num_workers = 1
    num_iterations = 20
    burnin_iterations = 5

    results = {}
    for backprop in (True, False):
        cpu_elapsed_times, cpu_memory_peaks = run_benchmark_cpu(
            model,
            data_loader,
            num_iterations=num_iterations,
            backprop=backprop,
            burnin_iterations=burnin_iterations,
        )
        results[f"CPU_time_backprop_{backprop}"] = sum(cpu_elapsed_times) / len(
            cpu_elapsed_times
        )
        results[f"CPU_memory_backprop_{backprop}"] = max(cpu_memory_peaks)

        gpu_elapsed_times, gpu_memory_peaks = run_benchmark_cuda(
            model,
            data_loader,
            num_iterations=num_iterations,
            backprop=backprop,
            burnin_iterations=burnin_iterations,
        )
        results[f"GPU_time_backprop_{backprop}"] = sum(gpu_elapsed_times) / len(
            gpu_elapsed_times
        )
        results[f"GPU_memory_backprop_{backprop}"] = max(gpu_memory_peaks)

    return results


def run_benchmark_cuda(
    model,
    data_loader: DataLoader,
    *,
    device: torch.device = torch.device("cuda"),
    num_iterations: int,
    burnin_iterations: int = 1,
    backprop: bool = False,
) -> tuple[list[float], list[float]]:
    """Adopted from: https://github.com/april-tools/sos-npcs/blob/8b3c8592ab1db852166c9d1cdf202c1cccc7e6d4/src/scripts/benchmarks/benchmark.py#L86"""

    import gc
    from torch import optim

    def infinite_dataloader():
        while True:
            yield from data_loader

    model = model.to(device)

    if backprop:
        # Setup losses and a dummy optimizer (only used to free gradient tensors)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    else:
        optimizer = None

    lambda_weight = 0.5  # example weighting factor
    elapsed_times = list()
    gpu_memory_peaks = list()
    for i, batch in enumerate(infinite_dataloader()):
        if i == num_iterations + burnin_iterations:
            break
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        # Run GC manually and then disable it
        gc.collect()
        gc.disable()
        # Reset peak memory usage statistics
        torch.cuda.reset_peak_memory_stats(device)

        batch = batch.to(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(device))

        # forward

        if backprop:
            y, c = model(batch)
            pseudo_target_c = torch.zeros_like(c, device=device)
            pseudo_target_y = torch.zeros_like(y, device=device)
            concept_loss = model.concept_loss_function(c, pseudo_target_c)
            task_loss = model.task_loss_function(y, pseudo_target_y)
            total_loss = task_loss + concept_loss * lambda_weight
            total_loss.backward(retain_graph=False)
        else:
            with torch.no_grad():
                y, c = model(batch)

        end.record(torch.cuda.current_stream(device))
        torch.cuda.synchronize(device)  # Synchronize CUDA Kernels before measuring time
        # end_time = time.perf_counter()
        gpu_memory_peaks.append(torch.cuda.max_memory_allocated(device) / (1024**2))

        if backprop:
            assert optimizer is not None
            optimizer.zero_grad()  # Free gradients tensors
        gc.enable()  # Enable GC again
        gc.collect()  # Manual GC
        # elapsed_times.append(end_time - start_time)
        elapsed_times.append(start.elapsed_time(end) * 1e-3)

    # Discard burnin iterations and compute averages
    elapsed_times = elapsed_times[burnin_iterations:]
    gpu_memory_peaks = gpu_memory_peaks[burnin_iterations:]
    return elapsed_times, gpu_memory_peaks


def run_benchmark_cpu(
    model: torch.nn.Module,
    data_loader: DataLoader,
    *,
    device: torch.device = torch.device("cpu"),
    num_iterations: int,
    burnin_iterations: int = 1,
    backprop: bool = False,
):
    import time
    import gc
    from torch import optim
    import tracemalloc

    def infinite_dataloader():
        while True:
            yield from data_loader

    model = model.to(device)

    if backprop:
        optimizer = optim.SGD(model.parameters(), lr=0.01)
    else:
        optimizer = None

    elapsed_times = []
    memory_peaks = []  # Placeholder for CPU memory usage (optional)
    lambda_weight = 0.5  # example weighting factor

    # process = psutil.Process(os.getpid())

    for i, batch in enumerate(infinite_dataloader()):
        if i == num_iterations + burnin_iterations:
            break
        if isinstance(batch, (tuple, list)):
            batch = batch[0]

        # Run GC manually and then disable it to reduce noise in timing
        gc.collect()
        gc.disable()

        batch = batch.to(device)

        # Start Python memory tracking
        tracemalloc.start()

        start_time = time.perf_counter()

        # forward
        if backprop:
            y, c = model(batch)
            pseudo_target_c = torch.zeros_like(c, device=device)
            pseudo_target_y = torch.zeros_like(y, device=device)
            concept_loss = model.concept_loss_function(c, pseudo_target_c)
            task_loss = model.task_loss_function(y, pseudo_target_y)
            total_loss = task_loss + concept_loss * lambda_weight
            total_loss.backward(retain_graph=False)
        else:
            with torch.no_grad():
                y, c = model(batch)
        end_time = time.perf_counter()

        # Measure memory usage after forward pass
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memory_peaks.append(peak_mem / (1024**2))  # Convert bytes → MB

        if backprop:
            assert optimizer is not None
            optimizer.zero_grad()

        # Cleanup
        del y, c, batch
        gc.enable()
        gc.collect()

        elapsed_times.append(end_time - start_time)

    # Discard burnin iterations
    elapsed_times = elapsed_times[burnin_iterations:]
    memory_peaks = memory_peaks[burnin_iterations:]

    return elapsed_times, memory_peaks
