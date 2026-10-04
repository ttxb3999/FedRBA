import os
import sys
import copy
import pickle
import random
import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from models import get_model
from .fairfal_sampling import FairFALSampling
from .kafal_sampling import KAFALSampling

from collections import Counter


class _SelectedImageDataset(Dataset):
    """Index-only dataset for prediction without retaining sample labels."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = [int(index) for index in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        image, _ = self.dataset[self.indices[position]]
        return image, position

def get_label_count(dataset, index_list, args):
    if args.dataset in ['pathmnist', 'octmnist', 'organamnist', 'dermamnist', 'bloodmnist']:
        labels = np.asarray(dataset.labels)[index_list].ravel().tolist()
    else:
        labels = np.asarray(dataset.targets)[index_list].tolist()
    return Counter(labels)


def _get_dataset_label(dataset, idx, args):
    """Read the currently observed label of one sample."""
    idx = int(idx)

    if args.dataset in [
        'pathmnist',
        'octmnist',
        'organamnist',
        'dermamnist',
        'bloodmnist'
    ]:
        value = np.asarray(dataset.labels)[idx]
        return int(np.asarray(value).reshape(-1)[0])

    return int(np.asarray(dataset.targets)[idx])


def _set_dataset_label(dataset, idx, new_label, args):
    """Overwrite one sample's observed label."""
    idx = int(idx)
    new_label = int(new_label)

    if args.dataset in [
        'pathmnist',
        'octmnist',
        'organamnist',
        'dermamnist',
        'bloodmnist'
    ]:
        labels = dataset.labels

        if np.asarray(labels).ndim == 2:
            labels[idx, 0] = new_label
        else:
            labels[idx] = new_label
    else:
        dataset.targets[idx] = new_label


def _get_annotation_noise_rate(user_idx, args):
    """
    Return queried-label error rate for this client.

    clean:
        all clients 0%

    uniform:
        all clients 19%

    hetero:
        clients 0-2: 0%
        clients 3-5: 10%
        clients 6-7: 30%
        clients 8-9: 50%
    """
    if args.noise_mode == 'clean':
        return 0.0

    if args.noise_mode == 'uniform':
        return float(args.uniform_noise_rate)

    if args.noise_mode == 'custom':
        return float(args.custom_noise_rates[int(user_idx)])

    if args.noise_mode == 'hetero':
        if args.num_users != 10:
            raise ValueError(
                'Current hetero noise profile requires num_users=10'
            )

        noise_profile = [
            0.0, 0.0, 0.0,
            0.1, 0.1, 0.1,
            0.3, 0.3,
            0.5, 0.5
        ]

        return noise_profile[int(user_idx)]

    raise ValueError(
        'Unknown noise mode: {}'.format(args.noise_mode)
    )


def apply_query_annotation_noise(
    dataset_train,
    dataset_query,
    new_data,
    user_idx,
    args
):
    """
    Corrupt labels ONLY for newly queried samples.

    A separate RNG is used so annotation noise does not
    consume the global NumPy random state used by training.
    """
    new_data = list(new_data)
    n_query = len(new_data)

    noise_rate = _get_annotation_noise_rate(
        user_idx,
        args
    )

    # Python round() gives the desired counts for n_query=85:
    # 10% -> 8, 30% -> 26, 50% -> 42.
    n_corrupt = int(round(n_query * noise_rate))

    # Independent deterministic RNG:
    # changing annotation noise will not alter the global
    # random sequence used by FL/client sampling.
    stage_id = int(round(args.current_ratio * 1000))

    noise_seed = (
        int(args.seed) * 1000003
        + stage_id * 1009
        + int(user_idx) * 97
        + 2027
    )

    rng = np.random.default_rng(noise_seed)

    if n_corrupt > 0:
        corrupt_positions = set(
            rng.choice(
                n_query,
                size=n_corrupt,
                replace=False
            ).tolist()
        )
    else:
        corrupt_positions = set()

    records = []

    for position, data_idx in enumerate(new_data):
        true_label = _get_dataset_label(
            dataset_query,
            data_idx,
            args
        )

        observed_label = true_label

        if position in corrupt_positions:
            # Randomly choose one of the OTHER classes.
            wrong_label = int(
                rng.integers(0, args.num_classes - 1)
            )

            if wrong_label >= true_label:
                wrong_label += 1

            observed_label = wrong_label

            # Training and querying must see the same
            # annotator-provided label.
            _set_dataset_label(
                dataset_train,
                data_idx,
                observed_label,
                args
            )

            _set_dataset_label(
                dataset_query,
                data_idx,
                observed_label,
                args
            )

            records.append(
                (
                    int(data_idx),
                    int(true_label),
                    int(observed_label)
                )
            )

    print(
        '[Annotation Noise] '
        'client {} | mode={} | rate={:.0%} | corrupted={}/{}'.format(
            user_idx,
            args.noise_mode,
            noise_rate,
            n_corrupt,
            n_query
        )
    )

    if records:
        print(
            'noise examples (idx, true, observed):',
            records[:8]
        )

    # Save one summary row for every client and one row per corruption.
    noise_log_path = os.path.join(
        args.result_dir,
        'annotation_noise.csv'
    )
    log_exists = os.path.exists(noise_log_path)
    with open(noise_log_path, 'a', encoding='utf-8') as f:
        if not log_exists:
            f.write(
                'current_ratio,user_idx,noise_mode,noise_rate,'
                'n_corrupt,n_query,sample_idx,true_label,observed_label\n'
            )

        # An empty sample_idx marks the per-client summary row. This also
        # records clean clients, for which there are no corruption rows.
        f.write(
            '{:.3f},{},{},{:.6f},{},{},,,\n'.format(
                args.current_ratio,
                user_idx,
                args.noise_mode,
                noise_rate,
                n_corrupt,
                n_query
            )
        )
        for data_idx, true_label, observed_label in records:
            f.write(
                '{:.3f},{},{},{:.6f},{},{},{},{},{}\n'.format(
                    args.current_ratio,
                    user_idx,
                    args.noise_mode,
                    noise_rate,
                    n_corrupt,
                    n_query,
                    data_idx,
                    true_label,
                    observed_label
                )
            )

    return {
        'noise_rate': noise_rate,
        'n_corrupt': n_corrupt,
        'n_query': n_query,
        'records': records
    }

def random_query_samples(dict_users_train_total, dict_users_test_total, args):
    """ randomly select the labeled samples at the first round
    """
    args.dict_users_total_path = os.path.join(args.dict_user_path, 'dict_users_train_test_total.pkl'.format(args.seed))
            
    with open(args.dict_users_total_path, 'wb') as handle:
        pickle.dump((dict_users_train_total, dict_users_test_total), handle)
        
    dict_users_train_label_path = os.path.join(args.dict_user_path, 'dict_users_train_label_{:.3f}.pkl'.format(args.current_ratio))

    dict_users_train_label = {user_idx: [] for user_idx in dict_users_train_total.keys()}

    # sample n_start example on each client
    for idx in dict_users_train_total.keys():
        dict_users_train_label[idx] = np.random.choice(np.array(list(dict_users_train_total[idx])), int(args.n_data / args.num_users), replace=False)
        
    if args.budget_mode == 'estimated_r_anchor_blend':
        # This immutable copy defines the guaranteed-clean initial-label
        # anchor. It is deliberately separate from later
        # dict_users_train_label_{ratio} files, which mix initial and actively
        # queried samples.
        anchor_path = os.path.join(
            args.dict_user_path, 'initial_clean_anchor_indices.pkl'
        )
        if os.path.exists(anchor_path):
            with open(anchor_path, 'rb') as handle:
                existing_anchor = pickle.load(handle)
            if set(existing_anchor.keys()) != set(dict_users_train_label.keys()):
                raise ValueError('existing clean-anchor clients do not match initialization')
            for user_idx, indices in dict_users_train_label.items():
                if not np.array_equal(
                    np.asarray(existing_anchor[user_idx]), np.asarray(indices)
                ):
                    raise ValueError(
                        'existing initial_clean_anchor_indices.pkl differs from '
                        'the newly sampled initial clean indices; refusing overwrite'
                    )
        else:
            immutable_anchor = {
                user_idx: np.asarray(indices).copy()
                for user_idx, indices in dict_users_train_label.items()
            }
            with open(anchor_path, 'wb') as handle:
                pickle.dump(immutable_anchor, handle)

    with open(dict_users_train_label_path, 'wb') as handle:
        pickle.dump(dict_users_train_label, handle)
    
    return dict_users_train_label, args


def _normalize_oracle_values(values):
    values = np.asarray(values, dtype=np.float64)
    mean_value = float(values.mean()) if len(values) else 0.0
    if mean_value > 0:
        return values / mean_value
    return np.ones_like(values)


def _allocate_oracle_budgets(user_ids, capacities, scores, total_budget, min_budget):
    """Allocate an exact integer budget without consuming any RNG state."""
    user_ids = list(user_ids)
    capacities = {user_idx: int(capacities[user_idx]) for user_idx in user_ids}
    scores = {user_idx: float(scores[user_idx]) for user_idx in user_ids}

    if total_budget < min_budget * len(user_ids):
        raise ValueError(
            'total query budget {} is smaller than {} clients times min budget {}'.format(
                total_budget, len(user_ids), min_budget
            )
        )
    if total_budget > sum(capacities.values()):
        raise ValueError(
            'global unlabeled capacity {} is smaller than total query budget {}'.format(
                sum(capacities.values()), total_budget
            )
        )

    # Capacity is a hard constraint; a client with fewer than min_budget
    # available examples receives all it has, then the remainder is reassigned.
    budgets = {
        user_idx: min(min_budget, capacities[user_idx])
        for user_idx in user_ids
    }
    remaining_budget = total_budget - sum(budgets.values())

    while remaining_budget > 0:
        eligible = [
            user_idx for user_idx in user_ids
            if budgets[user_idx] < capacities[user_idx]
        ]
        if not eligible:
            raise ValueError(
                'unable to allocate the remaining {} query budget units within client capacities'.format(
                    remaining_budget
                )
            )

        weight_sum = sum(max(scores[user_idx], 0.0) for user_idx in eligible)
        if weight_sum <= 0:
            weights = {user_idx: 1.0 for user_idx in eligible}
            weight_sum = float(len(eligible))
        else:
            weights = {
                user_idx: max(scores[user_idx], 0.0)
                for user_idx in eligible
            }

        shares = {
            user_idx: remaining_budget * weights[user_idx] / weight_sum
            for user_idx in eligible
        }
        floor_additions = {
            user_idx: min(
                int(np.floor(shares[user_idx])),
                capacities[user_idx] - budgets[user_idx]
            )
            for user_idx in eligible
        }
        allocated = sum(floor_additions.values())
        for user_idx, addition in floor_additions.items():
            budgets[user_idx] += addition
        remaining_budget -= allocated

        if remaining_budget == 0:
            break

        # Deterministic largest-remainder allocation. Equal fractional parts
        # are resolved by ascending user_idx.
        remainders = sorted(
            (
                (shares[user_idx] - np.floor(shares[user_idx]), user_idx)
                for user_idx in eligible
                if budgets[user_idx] < capacities[user_idx]
            ),
            key=lambda item: (-item[0], item[1])
        )
        if not remainders:
            continue

        allocated = 0
        for _, user_idx in remainders:
            if remaining_budget == 0:
                break
            budgets[user_idx] += 1
            remaining_budget -= 1
            allocated += 1
        if allocated == 0:
            raise ValueError('deterministic largest-remainder allocation made no progress')

    if sum(budgets.values()) != total_budget:
        raise ValueError('allocated budget does not equal total query budget')
    return budgets


def _predict_selected_global_metadata(query_net, dataset_query, new_data, args):
    """Predict selected samples with the shared global query model only."""
    selected_dataset = _SelectedImageDataset(dataset_query, new_data)
    generator = torch.Generator()
    generator.manual_seed(int(args.seed) * 1000003 + 7919)
    loader = DataLoader(
        selected_dataset,
        batch_size=max(int(args.test_bs), 1),
        shuffle=False,
        num_workers=0,
        generator=generator
    )

    predicted_labels = torch.empty(len(selected_dataset), dtype=torch.long)
    confidences = torch.empty(len(selected_dataset), dtype=torch.float32)
    was_training = query_net.training
    query_net.eval()
    try:
        with torch.no_grad():
            for images, positions in loader:
                images = images.to(args.device)
                logits, _ = query_net(images)
                probabilities = torch.softmax(logits, dim=1)
                confidence, prediction = probabilities.max(dim=1)
                predicted_labels[positions] = prediction.cpu()
                confidences[positions] = confidence.cpu()
    finally:
        query_net.train(was_training)

    return predicted_labels.numpy(), confidences.numpy()


def _predict_selected_global_probabilities(query_net, dataset_query, new_data, args):
    """Predict full softmax probabilities with the shared global query model."""
    selected_dataset = _SelectedImageDataset(dataset_query, new_data)
    generator = torch.Generator()
    generator.manual_seed(int(args.seed) * 1000003 + 7919)
    loader = DataLoader(
        selected_dataset,
        batch_size=max(int(args.test_bs), 1),
        shuffle=False,
        num_workers=0,
        generator=generator
    )

    probabilities = torch.empty(
        (len(selected_dataset), int(args.num_classes)),
        dtype=torch.float32
    )
    was_training = query_net.training
    query_net.eval()
    try:
        with torch.no_grad():
            for images, positions in loader:
                images = images.to(args.device)
                logits, _ = query_net(images)
                batch_probabilities = torch.softmax(logits, dim=1)
                probabilities[positions] = batch_probabilities.cpu()
    finally:
        query_net.train(was_training)

    return probabilities.numpy()


def _update_estimated_reliability(
    predicted_labels,
    confidences,
    observed_labels,
    previous_reliability,
    num_classes,
    threshold,
    beta,
    min_confident
):
    """Update reliability using only predictions, confidences, and observed labels."""
    predicted_labels = np.asarray(predicted_labels, dtype=np.int64).reshape(-1)
    confidences = np.asarray(confidences, dtype=np.float64).reshape(-1)
    observed_labels = np.asarray(observed_labels, dtype=np.int64).reshape(-1)

    if not (
        len(predicted_labels) == len(confidences) == len(observed_labels)
    ):
        raise ValueError('Estimated-R prediction and observed-label lengths must match')

    previous_reliability = float(np.clip(previous_reliability, 0.0, 1.0))
    confident_mask = confidences >= float(threshold)
    n_confident = int(confident_mask.sum())
    n_agree = int(
        (predicted_labels[confident_mask] == observed_labels[confident_mask]).sum()
    )

    agreement_rate = float('nan')
    mean_confidence = float('nan')
    cycle_reliability = float('nan')
    updated_reliability = previous_reliability

    if n_confident > 0:
        agreement_rate = n_agree / n_confident
        mean_confidence = float(confidences[confident_mask].mean())

    if n_confident >= int(min_confident):
        wrong_match_prob = (1.0 - mean_confidence) / (int(num_classes) - 1)
        denominator = mean_confidence - wrong_match_prob
        if denominator > 1e-12:
            estimated_error = (mean_confidence - agreement_rate) / denominator
            estimated_error = float(np.clip(estimated_error, 0.0, 1.0))
            cycle_reliability = 1.0 - estimated_error
            updated_reliability = (
                float(beta) * previous_reliability
                + (1.0 - float(beta)) * cycle_reliability
            )
            updated_reliability = float(np.clip(updated_reliability, 0.0, 1.0))

    return {
        'n_query': int(len(predicted_labels)),
        'n_confident': n_confident,
        'n_agree': n_agree,
        'agreement_rate': agreement_rate,
        'mean_confidence': mean_confidence,
        'cycle_reliability': cycle_reliability,
        'updated_reliability': updated_reliability
    }


def _update_soft_likelihood_reliability(
    probabilities,
    observed_labels,
    previous_reliability,
    num_classes,
    beta
):
    """Estimate reliability from soft predictions and observed labels only."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    observed_labels = np.asarray(observed_labels, dtype=np.int64).reshape(-1)
    num_classes = int(num_classes)

    if probabilities.ndim != 2:
        raise ValueError('Soft Estimated-R probabilities must be a 2D array')
    if probabilities.shape[0] != len(observed_labels):
        raise ValueError(
            'Soft Estimated-R probability and observed-label lengths must match'
        )
    if probabilities.shape[1] != num_classes:
        raise ValueError(
            'Soft Estimated-R probability columns must equal num_classes'
        )
    if num_classes < 2:
        raise ValueError('Soft Estimated-R requires at least two classes')
    if np.any((observed_labels < 0) | (observed_labels >= num_classes)):
        raise ValueError('Soft Estimated-R observed labels are out of range')

    previous_reliability = float(np.clip(previous_reliability, 0.0, 1.0))
    q_values = probabilities[np.arange(len(observed_labels)), observed_labels]
    e_grid = np.linspace(0.0, 1.0, 1001)
    likelihood_probabilities = (
        (1.0 - e_grid[:, None]) * q_values[None, :]
        + e_grid[:, None] / (num_classes - 1) * (1.0 - q_values[None, :])
    )
    log_likelihood_grid = np.log(
        np.clip(likelihood_probabilities, 1e-12, None)
    ).sum(axis=1)

    max_log_likelihood = float(log_likelihood_grid.max())
    min_log_likelihood = float(log_likelihood_grid.min())
    likelihood_range = max_log_likelihood - min_log_likelihood
    mean_observed_probability = (
        float(q_values.mean()) if len(q_values) > 0 else float('nan')
    )
    median_observed_probability = (
        float(np.median(q_values)) if len(q_values) > 0 else float('nan')
    )
    mean_max_confidence = (
        float(probabilities.max(axis=1).mean())
        if len(probabilities) > 0 else float('nan')
    )

    mle_error_rate = float('nan')
    cycle_reliability = float('nan')
    updated_reliability = previous_reliability
    mle_at_boundary = 0
    if likelihood_range > 1e-6:
        # np.argmax returns the first maximum, hence the smallest e on ties.
        mle_index = int(np.argmax(log_likelihood_grid))
        mle_error_rate = float(e_grid[mle_index])
        cycle_reliability = 1.0 - mle_error_rate
        updated_reliability = (
            float(beta) * previous_reliability
            + (1.0 - float(beta)) * cycle_reliability
        )
        updated_reliability = float(np.clip(updated_reliability, 0.0, 1.0))
        mle_at_boundary = int(mle_index == 0 or mle_index == len(e_grid) - 1)

    return {
        'n_query': int(len(observed_labels)),
        'mean_observed_probability': mean_observed_probability,
        'median_observed_probability': median_observed_probability,
        'mean_max_confidence': mean_max_confidence,
        'mle_error_rate': mle_error_rate,
        'cycle_reliability': cycle_reliability,
        'updated_reliability': updated_reliability,
        'max_log_likelihood': max_log_likelihood,
        'min_log_likelihood': min_log_likelihood,
        'likelihood_range': likelihood_range,
        'mle_at_boundary': mle_at_boundary
    }


def _estimate_soft_likelihood_error(probabilities, observed_labels, num_classes):
    """Pure deterministic Soft-MLE estimate, with no EMA/state mutation."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    observed_labels = np.asarray(observed_labels, dtype=np.int64).reshape(-1)
    num_classes = int(num_classes)

    if probabilities.ndim != 2:
        raise ValueError('Soft likelihood probabilities must be a 2D array')
    if probabilities.shape[0] != len(observed_labels):
        raise ValueError('Soft likelihood probability and label lengths must match')
    if probabilities.shape[1] != num_classes or num_classes < 2:
        raise ValueError('Soft likelihood requires num_classes matching probability columns')
    if np.any((observed_labels < 0) | (observed_labels >= num_classes)):
        raise ValueError('Soft likelihood observed labels are out of range')

    # One point cannot provide a stable client-level mismatch baseline.
    if len(observed_labels) < 2:
        return {
            'n_samples': int(len(observed_labels)), 'mle_error_rate': float('nan'),
            'likelihood_range': float('nan'), 'mean_observed_probability': float('nan'),
            'mean_max_confidence': float('nan'), 'boundary_flag': 0,
            'valid': False, 'skip_reason': 'insufficient_samples'
        }

    q_values = probabilities[np.arange(len(observed_labels)), observed_labels]
    e_grid = np.linspace(0.0, 1.0, 1001)
    likelihood_probabilities = (
        (1.0 - e_grid[:, None]) * q_values[None, :]
        + e_grid[:, None] / (num_classes - 1) * (1.0 - q_values[None, :])
    )
    log_likelihood_grid = np.log(
        np.clip(likelihood_probabilities, 1e-12, None)
    ).sum(axis=1)
    likelihood_range = float(log_likelihood_grid.max() - log_likelihood_grid.min())
    mle_index = int(np.argmax(log_likelihood_grid))
    valid = likelihood_range > 1e-6
    return {
        'n_samples': int(len(observed_labels)),
        'mle_error_rate': float(e_grid[mle_index]) if valid else float('nan'),
        'likelihood_range': likelihood_range,
        'mean_observed_probability': float(q_values.mean()),
        'mean_max_confidence': float(probabilities.max(axis=1).mean()),
        'boundary_flag': int(mle_index == 0 or mle_index == len(e_grid) - 1),
        'valid': valid,
        'skip_reason': '' if valid else 'flat_likelihood'
    }


def _write_anchor_reliability_estimation_log(result_dir, current_ratio, record):
    """Write anchor-calibrated diagnostics; debug truth is logging-only."""
    log_path = os.path.join(result_dir, 'reliability_anchor_estimation.csv')
    log_exists = os.path.exists(log_path)
    fields = [
        'current_ratio', 'user_idx', 'budget_mode', 'allocated_budget',
        'n_query', 'n_anchor', 'e_query', 'e_anchor', 'denominator',
        'e_ann_raw', 'e_ann_calibrated', 'query_likelihood_range',
        'anchor_likelihood_range', 'query_mean_p_observed',
        'anchor_mean_p_observed', 'query_mean_max_confidence',
        'anchor_mean_max_confidence', 'query_boundary_flag',
        'anchor_boundary_flag', 'calibration_boundary', 'cycle_reliability',
        'previous_reliability', 'updated_reliability', 'valid', 'skip_reason',
        'true_reliability_debug', 'prediction_source'
    ]
    with open(log_path, 'a', encoding='utf-8') as f:
        if not log_exists:
            f.write(','.join(fields) + '\n')
        values = [
            '{:.3f}'.format(current_ratio), record['user_idx'], record['budget_mode'],
            record['allocated_budget'], record['n_query'], record['n_anchor'],
            record['e_query'], record['e_anchor'], record['denominator'],
            record['e_ann_raw'], record['e_ann_calibrated'],
            record['query_likelihood_range'], record['anchor_likelihood_range'],
            record['query_mean_p_observed'], record['anchor_mean_p_observed'],
            record['query_mean_max_confidence'], record['anchor_mean_max_confidence'],
            record['query_boundary_flag'], record['anchor_boundary_flag'],
            record['calibration_boundary'], record['cycle_reliability'],
            record['previous_reliability'], record['updated_reliability'],
            int(record['valid']), record['skip_reason'],
            record['true_reliability_debug'], record['prediction_source']
        ]
        f.write(','.join(str(value) for value in values) + '\n')


def _rankdata_average(values):
    """Average-tie ranks without a scipy dependency."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _pearson_correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2:
        return float('nan')
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = np.sqrt((left_centered ** 2).sum() * (right_centered ** 2).sum())
    return float(np.dot(left_centered, right_centered) / denominator) if denominator > 0 else float('nan')


def _print_anchor_reliability_diagnostic(reliability_state, true_reliability_debug):
    user_ids = sorted(reliability_state.keys())
    estimated = [float(reliability_state[user_idx]) for user_idx in user_ids]
    truth = [float(true_reliability_debug[user_idx]) for user_idx in user_ids]
    spearman = _pearson_correlation(_rankdata_average(estimated), _rankdata_average(truth))
    pearson = _pearson_correlation(estimated, truth)
    print('[Anchor Reliability Diagnostic]')
    print('estimated_reliability={}'.format(estimated))
    print('true_reliability_debug={}'.format(truth))
    print('Spearman={:.6f}'.format(spearman))
    print('Pearson={:.6f}'.format(pearson))
    for target, clients in [(1.0, [0, 1, 2]), (0.9, [3, 4, 5]), (0.7, [6, 7]), (0.5, [8, 9])]:
        available = [client for client in clients if client in reliability_state]
        if available:
            mean_estimated = float(np.mean([reliability_state[client] for client in available]))
            print('true R={} clients {}: estimated mean {:.6f}'.format(target, available, mean_estimated))


def _get_estimated_budget_allocation(
    dict_users_train_total,
    dict_users_train_label,
    reliability_state,
    args
):
    """Allocate Estimated-R budgets without reading labels or noise rates."""
    user_ids = sorted(dict_users_train_total.keys())
    base_budget = int(args.n_query / args.num_users)
    total_budget = base_budget * args.num_users
    min_budget = int(np.floor(base_budget * args.budget_min_ratio))
    capacities = {}
    for user_idx in user_ids:
        capacities[user_idx] = len(
            set(dict_users_train_total[user_idx])
            - set(dict_users_train_label[user_idx])
        )

    reliability_values = [reliability_state[user_idx] for user_idx in user_ids]
    normalized_values = _normalize_oracle_values(reliability_values)
    scores = {
        user_idx: float(normalized_values[position])
        for position, user_idx in enumerate(user_ids)
    }
    budgets = _allocate_oracle_budgets(
        user_ids, capacities, scores, total_budget, min_budget
    )
    return budgets, scores, total_budget


def _get_fixed_budget_allocation(user_ids, capacities, total_budget):
    """Build an exact, capacity-aware fixed reference allocation."""
    return _allocate_oracle_budgets(
        sorted(user_ids), capacities,
        {user_idx: 1.0 for user_idx in user_ids}, total_budget, 0
    )


def _blend_query_budgets(
    user_ids,
    fixed_budgets,
    reliability_budgets,
    capacities,
    blend_lambda,
    total_budget
):
    """Deterministically round a capacity-constrained fixed/reliability blend."""
    if not 0.0 <= blend_lambda <= 1.0:
        raise ValueError('blend_lambda must be within [0, 1]')

    user_ids = sorted(user_ids)
    capacities = {user_idx: int(capacities[user_idx]) for user_idx in user_ids}
    if total_budget > sum(capacities.values()):
        raise ValueError('global unlabeled capacity is smaller than total query budget')

    targets = {
        user_idx: (
            (1.0 - blend_lambda) * float(fixed_budgets[user_idx])
            + blend_lambda * float(reliability_budgets[user_idx])
        )
        for user_idx in user_ids
    }
    budgets = {
        user_idx: min(int(np.floor(targets[user_idx])), capacities[user_idx])
        for user_idx in user_ids
    }
    remaining_budget = int(total_budget - sum(budgets.values()))

    # Largest remainder, then ascending user_idx. If a capacity prevents an
    # initial rounding choice, continue through that same deterministic order.
    remainder_order = sorted(
        (
            (targets[user_idx] - np.floor(targets[user_idx]), user_idx)
            for user_idx in user_ids
        ),
        key=lambda item: (-item[0], item[1])
    )
    while remaining_budget > 0:
        allocated = 0
        for _, user_idx in remainder_order:
            if remaining_budget == 0:
                break
            if budgets[user_idx] < capacities[user_idx]:
                budgets[user_idx] += 1
                remaining_budget -= 1
                allocated += 1
        if allocated == 0:
            raise ValueError('unable to redistribute blended budget within client capacities')

    if sum(budgets.values()) != total_budget:
        raise ValueError('blended budget does not equal total query budget')
    return budgets


def _write_reliability_estimation_log(result_dir, current_ratio, record):
    """Write Estimated-R diagnostics; debug ground truth is logging-only."""
    log_path = os.path.join(result_dir, 'reliability_estimation.csv')
    log_exists = os.path.exists(log_path)
    with open(log_path, 'a', encoding='utf-8') as f:
        if not log_exists:
            f.write(
                'current_ratio,user_idx,budget_mode,allocated_budget,n_query,'
                'n_confident,n_agree,agreement_rate,mean_confidence,'
                'previous_reliability,cycle_reliability,updated_reliability,'
                'true_noise_rate_debug,true_reliability_debug,prediction_source\n'
            )
        f.write(
            '{:.3f},{},{},{},{},{},{},{:.6f},{:.6f},{:.6f},{:.6f},'
            '{:.6f},{:.6f},{:.6f},{}\n'.format(
                current_ratio,
                record['user_idx'],
                record['budget_mode'],
                record['allocated_budget'],
                record['n_query'],
                record['n_confident'],
                record['n_agree'],
                record['agreement_rate'],
                record['mean_confidence'],
                record['previous_reliability'],
                record['cycle_reliability'],
                record['updated_reliability'],
                record['true_noise_rate_debug'],
                record['true_reliability_debug'],
                record['prediction_source']
            )
        )


def _write_soft_reliability_estimation_log(result_dir, current_ratio, record):
    """Write soft Estimated-R diagnostics; debug truth is logging-only."""
    log_path = os.path.join(result_dir, 'reliability_soft_estimation.csv')
    log_exists = os.path.exists(log_path)
    with open(log_path, 'a', encoding='utf-8') as f:
        if not log_exists:
            f.write(
                'current_ratio,user_idx,budget_mode,allocated_budget,n_query,'
                'mean_observed_probability,median_observed_probability,'
                'mean_max_confidence,mle_error_rate,cycle_reliability,'
                'previous_reliability,updated_reliability,max_log_likelihood,'
                'min_log_likelihood,likelihood_range,mle_at_boundary,'
                'true_noise_rate_debug,true_reliability_debug,prediction_source\n'
            )
        f.write(
            '{:.3f},{},{},{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},'
            '{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{},{:.6f},{:.6f},{}\n'.format(
                current_ratio,
                record['user_idx'],
                record['budget_mode'],
                record['allocated_budget'],
                record['n_query'],
                record['mean_observed_probability'],
                record['median_observed_probability'],
                record['mean_max_confidence'],
                record['mle_error_rate'],
                record['cycle_reliability'],
                record['previous_reliability'],
                record['updated_reliability'],
                record['max_log_likelihood'],
                record['min_log_likelihood'],
                record['likelihood_range'],
                record['mle_at_boundary'],
                record['true_noise_rate_debug'],
                record['true_reliability_debug'],
                record['prediction_source']
            )
        )


def _get_oracle_budget_allocation(
    dataset_query,
    dict_users_train_total,
    dict_users_train_label,
    args
):
    """Build Oracle diagnostic budgets before any query selection occurs.

    ORACLE DIAGNOSTIC ONLY: true annotation-noise rates and true unlabeled
    labels used here are NOT AVAILABLE TO THE REAL METHOD.
    """
    user_ids = sorted(dict_users_train_total.keys())
    base_budget = int(args.n_query / args.num_users)
    # Match the real Fixed/FedRAL implementation exactly. This is crucial for
    # CIFAR100-LT, where args.n_query=795 but Fixed queries 79 x 10 = 790.
    total_budget = base_budget * args.num_users
    min_budget = int(np.floor(base_budget * args.budget_min_ratio))
    capacities = {}
    use_oracle_reliability = args.budget_mode in [
        'oracle_r', 'oracle_ru', 'oracle_r_blend'
    ]
    use_oracle_utility = args.budget_mode in ['oracle_u', 'oracle_ru']
    unlabeled_counts = {} if use_oracle_utility else None

    for user_idx in user_ids:
        total_idxs = dict_users_train_total[user_idx]
        label_idxs = dict_users_train_label[user_idx]
        unlabel_idxs = list(set(total_idxs) - set(label_idxs))
        capacities[user_idx] = len(unlabel_idxs)
        if use_oracle_utility:
            # ORACLE DIAGNOSTIC ONLY: this reads true labels of the unlabeled pool.
            unlabeled_counts[user_idx] = get_label_count(dataset_query, unlabel_idxs, args)

    if use_oracle_utility:
        global_unlabeled_counts = Counter()
        for count in unlabeled_counts.values():
            global_unlabeled_counts.update(count)
        tail_classes = sorted(
            range(args.num_classes),
            key=lambda class_idx: (global_unlabeled_counts[class_idx], class_idx)
        )[:args.num_classes // 2]

    reliability = {}
    utility = {}
    for user_idx in user_ids:
        if use_oracle_reliability:
            # ORACLE DIAGNOSTIC ONLY: not available to the real method.
            reliability[user_idx] = 1.0 - _get_annotation_noise_rate(user_idx, args)
        else:
            reliability[user_idx] = float('nan')

        if not use_oracle_utility:
            utility[user_idx] = float('nan')
        elif capacities[user_idx] == 0:
            utility[user_idx] = 0.0
        else:
            tail_count = sum(unlabeled_counts[user_idx][class_idx] for class_idx in tail_classes)
            utility[user_idx] = tail_count / capacities[user_idx]

    if args.budget_mode in ['oracle_r', 'oracle_r_blend']:
        score_values = _normalize_oracle_values(
            [reliability[user_idx] for user_idx in user_ids]
        )
    elif args.budget_mode == 'oracle_u':
        score_values = _normalize_oracle_values(
            [utility[user_idx] for user_idx in user_ids]
        )
    elif args.budget_mode == 'oracle_ru':
        reliability_norm = _normalize_oracle_values(
            [reliability[user_idx] for user_idx in user_ids]
        )
        utility_norm = _normalize_oracle_values(
            [utility[user_idx] for user_idx in user_ids]
        )
        score_values = 0.5 * reliability_norm + 0.5 * utility_norm
    else:
        raise ValueError('oracle allocation requested for non-oracle budget mode {}'.format(args.budget_mode))

    scores = {
        user_idx: float(score_values[position])
        for position, user_idx in enumerate(user_ids)
    }
    if all(score <= 0 for score in scores.values()):
        scores = {user_idx: 1.0 for user_idx in user_ids}

    budgets = _allocate_oracle_budgets(
        user_ids, capacities, scores, total_budget, min_budget
    )
    noise_rates = {
        user_idx: 1.0 - reliability[user_idx]
        if use_oracle_reliability else float('nan')
        for user_idx in user_ids
    }
    return budgets, noise_rates, reliability, utility, scores, total_budget


def _get_shuffled_reliability_view(reliability_state, args):
    """Return a deterministic client-permuted view of Soft reliability.

    The estimator state itself remains indexed by the original client IDs.
    A local NumPy Generator is used so this negative control does not consume
    the global RNG stream used by training or annotation noise.
    """
    user_ids = sorted(reliability_state.keys())
    original = np.asarray(
        [float(reliability_state[user_idx]) for user_idx in user_ids],
        dtype=float
    )

    previous_ratio = float(args.current_ratio - args.query_ratio)
    first_acquisition = np.isclose(
        previous_ratio, float(args.query_ratio), rtol=0.0, atol=1e-12
    )
    if first_acquisition:
        perm_positions = np.arange(len(user_ids), dtype=np.int64)
    else:
        stage_id = int(round(float(args.current_ratio) * 1000))
        shuffle_seed = 10000019 + 1009 * int(args.seed) + 9176 * stage_id
        rng = np.random.default_rng(shuffle_seed)
        perm_positions = rng.permutation(len(user_ids))
        if np.array_equal(perm_positions, np.arange(len(user_ids))):
            perm_positions = np.roll(perm_positions, 1)

    shuffled = {
        user_idx: float(original[perm_positions[position]])
        for position, user_idx in enumerate(user_ids)
    }
    source_user = {
        user_idx: int(user_ids[int(perm_positions[position])])
        for position, user_idx in enumerate(user_ids)
    }

    if sorted(shuffled.values()) != sorted(float(v) for v in original):
        raise ValueError('Shuffled-R did not preserve the reliability-score multiset')

    return shuffled, source_user


def _write_shuffled_reliability_map(
    result_dir,
    current_ratio,
    user_ids,
    reliability_state,
    allocation_reliability,
    source_user
):
    """Log the original-to-allocation reliability mapping for Shuffled-R."""
    log_path = os.path.join(result_dir, 'shuffled_reliability_map.csv')
    log_exists = os.path.exists(log_path)
    with open(log_path, 'a', encoding='utf-8') as f:
        if not log_exists:
            f.write(
                'current_ratio,user_idx,source_user_idx,'
                'original_reliability,allocation_reliability\n'
            )
        for user_idx in user_ids:
            f.write(
                '{:.3f},{},{},{:.9f},{:.9f}\n'.format(
                    current_ratio,
                    user_idx,
                    source_user[user_idx],
                    float(reliability_state[user_idx]),
                    float(allocation_reliability[user_idx])
                )
            )


def _write_budget_allocation_log(
    result_dir,
    current_ratio,
    budget_mode,
    user_ids,
    budgets,
    total_budget,
    noise_rates=None,
    reliability=None,
    utility=None,
    scores=None,
    fixed_budgets=None,
    raw_reliability_budgets=None,
    blend_lambda=None
):
    log_path = os.path.join(result_dir, 'budget_allocation.csv')
    log_exists = os.path.exists(log_path)
    with open(log_path, 'a', encoding='utf-8') as f:
        if not log_exists:
            f.write(
                'current_ratio,user_idx,budget_mode,allocated_budget,total_budget,'
                'noise_rate_oracle,reliability_oracle,utility_oracle,score,'
                'fixed_budget,raw_reliability_budget,blend_lambda\n'
            )
        for user_idx in user_ids:
            if noise_rates is None:
                oracle_fields = ',,,'
            else:
                oracle_fields = '{:.6f},{:.6f},{:.6f},{:.6f}'.format(
                    noise_rates[user_idx],
                    reliability[user_idx],
                    utility[user_idx],
                    scores[user_idx]
                )
            blend_fields = '{},{},{}'.format(
                'nan' if fixed_budgets is None else fixed_budgets[user_idx],
                'nan' if raw_reliability_budgets is None else raw_reliability_budgets[user_idx],
                'nan' if blend_lambda is None else blend_lambda
            )
            f.write(
                '{:.3f},{},{},{},{},{},{}\n'.format(
                    current_ratio,
                    user_idx,
                    budget_mode,
                    budgets[user_idx],
                    total_budget,
                    oracle_fields,
                    blend_fields
                )
            )
    
    
def algo_query_samples(
    dataset_train,
    dataset_query,
    dataset_test,
    dict_users_train_total,
    dict_users_test_total,
    mean_global_ratio,
    args,
    reliability_state=None
):
    """ query samples from the unlabeled pool
    """
    previous_ratio = args.current_ratio - args.query_ratio
    path = os.path.join(args.dict_user_path, 'dict_users_train_label_{:.3f}.pkl'.format(previous_ratio))    
    with open(path, 'rb') as f:
        dict_users_train_label = pickle.load(f) 

        print("Before Querying")
        total_data_cnt = 0
        for user_idx in range(args.num_users):
            print(user_idx, len(dict_users_train_label[user_idx]))
            total_data_cnt += len(dict_users_train_label[user_idx])

        print(total_data_cnt)
        print("-" * 20)

    initial_clean_anchor_indices = None
    if args.budget_mode == 'estimated_r_anchor_blend':
        anchor_path = os.path.join(
            args.dict_user_path, 'initial_clean_anchor_indices.pkl'
        )
        if not os.path.exists(anchor_path):
            raise FileNotFoundError(
                'estimated_r_anchor_blend requires {}'.format(anchor_path)
            )
        with open(anchor_path, 'rb') as handle:
            initial_clean_anchor_indices = pickle.load(handle)
        expected_users = set(dict_users_train_total.keys())
        if set(initial_clean_anchor_indices.keys()) != expected_users:
            raise ValueError('clean-anchor clients do not match active clients')
        initial_clean_anchor_indices = {
            user_idx: np.asarray(indices, dtype=np.int64).copy()
            for user_idx, indices in initial_clean_anchor_indices.items()
        }

    # Build model
    query_net = get_model(args)
    args.raw_ckpt = copy.deepcopy(query_net.state_dict())

    query_net_state_dict = torch.load(args.query_model)
    query_net.load_state_dict(query_net_state_dict)            
    
    # Methods
    if args.al_method == "fairfal":
        strategy = FairFALSampling(
            dataset_query,
            dataset_train,
            dataset_test,
            query_net,
            args
        )

    elif args.al_method == "kafal":
        strategy = KAFALSampling(
            dataset_query,
            dataset_train,
            dataset_test,
            query_net,
            args
        )

    else:
        exit('There is no al methods')   
    
    results_save_path = os.path.join(args.result_dir, 'query_info.txt')

    user_ids = sorted(dict_users_train_total.keys())
    estimated_reliability_before = None
    if args.budget_mode == 'fixed':
        # Preserve the original fixed-budget expression and do not invoke
        # any Oracle score/allocation code in the baseline path.
        fixed_budget = int(args.n_query / args.num_users)
        budgets = {user_idx: fixed_budget for user_idx in user_ids}
        total_budget = int(round(args.n_query))
        oracle_values = None
    elif args.budget_mode in ['estimated_r', 'estimated_r_soft']:
        if reliability_state is None:
            raise ValueError('Estimated-R requires an initialized reliability state')
        budgets, estimated_scores, total_budget = _get_estimated_budget_allocation(
            dict_users_train_total,
            dict_users_train_label,
            reliability_state,
            args
        )
        estimated_reliability_before = {
            user_idx: float(reliability_state[user_idx])
            for user_idx in user_ids
        }
        oracle_values = None
    elif args.budget_mode in [
        'estimated_r_soft_blend',
        'estimated_r_anchor_blend',
        'shuffled_r_soft_blend'
    ]:
        if reliability_state is None:
            raise ValueError('Estimated-R Soft Blend requires an initialized reliability state')

        allocation_reliability = reliability_state
        shuffled_source_user = None
        if args.budget_mode == 'shuffled_r_soft_blend':
            allocation_reliability, shuffled_source_user = _get_shuffled_reliability_view(
                reliability_state, args
            )

        raw_reliability_budgets, estimated_scores, total_budget = _get_estimated_budget_allocation(
            dict_users_train_total,
            dict_users_train_label,
            allocation_reliability,
            args
        )
        capacities = {
            user_idx: len(
                set(dict_users_train_total[user_idx])
                - set(dict_users_train_label[user_idx])
            )
            for user_idx in user_ids
        }
        fixed_budgets = _get_fixed_budget_allocation(user_ids, capacities, total_budget)
        budgets = _blend_query_budgets(
            user_ids,
            fixed_budgets,
            raw_reliability_budgets,
            capacities,
            args.reliability_budget_blend,
            total_budget
        )
        estimated_reliability_before = {
            user_idx: float(reliability_state[user_idx])
            for user_idx in user_ids
        }
        if args.budget_mode == 'shuffled_r_soft_blend':
            _write_shuffled_reliability_map(
                args.result_dir,
                args.current_ratio,
                user_ids,
                reliability_state,
                allocation_reliability,
                shuffled_source_user
            )
        oracle_values = None
    elif args.budget_mode == 'oracle_r_blend':
        (
            raw_oracle_r_budgets,
            noise_rates,
            reliability,
            utility,
            scores,
            total_budget
        ) = _get_oracle_budget_allocation(
            dataset_query,
            dict_users_train_total,
            dict_users_train_label,
            args
        )
        capacities = {
            user_idx: len(
                set(dict_users_train_total[user_idx])
                - set(dict_users_train_label[user_idx])
            )
            for user_idx in user_ids
        }
        fixed_budgets = _get_fixed_budget_allocation(user_ids, capacities, total_budget)

        # Warm-start-matched Oracle-R: the real method has no reliability
        # evidence before the first queried annotations arrive. Keep the first
        # acquisition (5% -> 10%) uniform; use true reliability only thereafter.
        previous_ratio = float(args.current_ratio - args.query_ratio)
        first_acquisition = np.isclose(
            previous_ratio, float(args.query_ratio), rtol=0.0, atol=1e-12
        )
        oracle_budgets_for_blend = (
            fixed_budgets if first_acquisition else raw_oracle_r_budgets
        )
        budgets = _blend_query_budgets(
            user_ids,
            fixed_budgets,
            oracle_budgets_for_blend,
            capacities,
            args.reliability_budget_blend,
            total_budget
        )
        if first_acquisition:
            raw_oracle_r_budgets = dict(fixed_budgets)
        oracle_values = (noise_rates, reliability, utility, scores)
    else:
        (
            budgets,
            noise_rates,
            reliability,
            utility,
            scores,
            total_budget
        ) = _get_oracle_budget_allocation(
            dataset_query,
            dict_users_train_total,
            dict_users_train_label,
            args
        )
        oracle_values = (noise_rates, reliability, utility, scores)

    print('[Budget Allocation]')
    print('mode={}'.format(args.budget_mode))
    if args.budget_mode in ['estimated_r_soft_blend', 'estimated_r_anchor_blend', 'shuffled_r_soft_blend', 'oracle_r_blend']:
        print('blend_lambda={}'.format(args.reliability_budget_blend))
        print('fixed_budgets={}'.format([fixed_budgets[user_idx] for user_idx in user_ids]))
        raw_budget_label = (
            'raw_oracle_r_budgets'
            if args.budget_mode == 'oracle_r_blend'
            else 'raw_reliability_budgets'
        )
        raw_budgets = (
            raw_oracle_r_budgets
            if args.budget_mode == 'oracle_r_blend'
            else raw_reliability_budgets
        )
        print('{}={}'.format(
            raw_budget_label,
            [raw_budgets[user_idx] for user_idx in user_ids]
        ))
    print('budgets={}'.format([budgets[user_idx] for user_idx in user_ids]))
    print('sum={}'.format(sum(budgets.values())))
    if estimated_reliability_before is not None:
        print(
            'estimated_reliability_before={}'.format(
                [estimated_reliability_before[user_idx] for user_idx in user_ids]
            )
        )
    if oracle_values is not None:
        _, reliability, utility, scores = oracle_values
        print('reliability={}'.format([reliability[user_idx] for user_idx in user_ids]))
        print('utility={}'.format([utility[user_idx] for user_idx in user_ids]))
        print('score={}'.format([scores[user_idx] for user_idx in user_ids]))

    if oracle_values is None:
        _write_budget_allocation_log(
            args.result_dir, args.current_ratio, args.budget_mode, user_ids,
            budgets, total_budget,
            fixed_budgets=(
                fixed_budgets
                if args.budget_mode in ['estimated_r_soft_blend', 'estimated_r_anchor_blend', 'shuffled_r_soft_blend'] else None
            ),
            raw_reliability_budgets=(
                raw_reliability_budgets
                if args.budget_mode in ['estimated_r_soft_blend', 'estimated_r_anchor_blend', 'shuffled_r_soft_blend'] else None
            ),
            blend_lambda=(
                args.reliability_budget_blend
                if args.budget_mode in ['estimated_r_soft_blend', 'estimated_r_anchor_blend', 'shuffled_r_soft_blend'] else None
            )
        )
    else:
        _write_budget_allocation_log(
            args.result_dir, args.current_ratio, args.budget_mode, user_ids,
            budgets, total_budget, *oracle_values,
            fixed_budgets=(
                fixed_budgets
                if args.budget_mode == 'oracle_r_blend' else None
            ),
            raw_reliability_budgets=(
                raw_oracle_r_budgets
                if args.budget_mode == 'oracle_r_blend' else None
            ),
            blend_lambda=(
                args.reliability_budget_blend
                if args.budget_mode == 'oracle_r_blend' else None
            )
        )

    time = datetime.timedelta()
    anchor_true_reliability_debug = {}
    sum_global_ratio, count = 0., 0.
    for user_idx in dict_users_train_total.keys():                
        total_idxs = dict_users_train_total[user_idx]
        label_idxs = dict_users_train_label[user_idx]
        unlabel_idxs = list(set(total_idxs) - set(label_idxs))
        
        count_label = get_label_count(dataset_query, label_idxs, args)
        global_ratio = strategy.query_model_selection(user_idx, label_idxs, unlabel_idxs, count_label)
        
        sum_global_ratio += global_ratio
        count += 1
    
    if mean_global_ratio == None:
        mean_global_ratio = sum_global_ratio / count
    
    print("mean_global_ratio: ", mean_global_ratio)    
    
    for user_idx in dict_users_train_total.keys():                
        total_idxs = dict_users_train_total[user_idx]
        label_idxs = dict_users_train_label[user_idx]
        unlabel_idxs = list(set(total_idxs) - set(label_idxs))
        
        test_idxs = dict_users_test_total[user_idx]
        
        start = datetime.datetime.now()
        count_label = get_label_count(dataset_query, label_idxs, args)
        count_unlabel = None
        if args.budget_mode not in ['estimated_r', 'estimated_r_soft', 'estimated_r_soft_blend', 'estimated_r_anchor_blend', 'shuffled_r_soft_blend']:
            count_unlabel = get_label_count(dataset_query, unlabel_idxs, args)
        print("user_idx is ", user_idx)
        print("count_label: ", count_label)
        if count_unlabel is not None:
            print("count_unlabel: ", count_unlabel)
        if args.budget_mode == 'fixed':
            new_data, global_acc, local_acc = strategy.query(
                user_idx,
                label_idxs,
                unlabel_idxs,
                test_idxs,
                count_label,
                mean_global_ratio,
                int(args.n_query / args.num_users)
            )
        else:
            new_data, global_acc, local_acc = strategy.query(
            user_idx,
            label_idxs,
            unlabel_idxs,
            test_idxs,
            count_label,
            mean_global_ratio,
                budgets[user_idx]
            )

        predicted_labels = None
        confidences = None
        probabilities = None
        if args.budget_mode == 'estimated_r':
            predicted_labels, confidences = _predict_selected_global_metadata(
                query_net,
                dataset_query,
                new_data,
                args
            )
        elif args.budget_mode in ['estimated_r_soft', 'estimated_r_soft_blend', 'estimated_r_anchor_blend', 'shuffled_r_soft_blend']:
            probabilities = _predict_selected_global_probabilities(
                query_net,
                dataset_query,
                new_data,
                args
            )

        anchor_probabilities = None
        anchor_observed_labels = None
        if args.budget_mode == 'estimated_r_anchor_blend':
            anchor_indices = initial_clean_anchor_indices[user_idx]
            anchor_probabilities = _predict_selected_global_probabilities(
                query_net,
                dataset_query,
                anchor_indices,
                args
            )
            # These labels are the fixed, guaranteed-clean initial labels.
            anchor_observed_labels = np.asarray(
                [_get_dataset_label(dataset_query, data_idx, args) for data_idx in anchor_indices],
                dtype=np.int64
            )

        count_query_true = None
        if args.budget_mode not in ['estimated_r', 'estimated_r_soft', 'estimated_r_soft_blend', 'estimated_r_anchor_blend', 'shuffled_r_soft_blend']:
            # Ground-truth class distribution BEFORE annotation noise.
            count_query_true = get_label_count(
                dataset_query,
                new_data,
                args
            )

        noise_stats = apply_query_annotation_noise(
            dataset_train,
            dataset_query,
            new_data,
            user_idx,
            args
        )

        if args.budget_mode == 'estimated_r':
            observed_labels = np.asarray(
                [
                    _get_dataset_label(dataset_query, data_idx, args)
                    for data_idx in new_data
                ],
                dtype=np.int64
            )
            previous_reliability = reliability_state[user_idx]
            reliability_update = _update_estimated_reliability(
                predicted_labels,
                confidences,
                observed_labels,
                previous_reliability,
                args.num_classes,
                args.reliability_threshold,
                args.reliability_ema_beta,
                args.reliability_min_confident
            )
            reliability_state[user_idx] = reliability_update['updated_reliability']

            # DEBUG/EVALUATION ONLY. NEVER USED BY ESTIMATED-R estimation,
            # allocation, or reliability state.
            true_noise_rate_debug = _get_annotation_noise_rate(user_idx, args)
            true_reliability_debug = 1.0 - true_noise_rate_debug
            _write_reliability_estimation_log(
                args.result_dir,
                args.current_ratio,
                {
                    'user_idx': user_idx,
                    'budget_mode': args.budget_mode,
                    'allocated_budget': budgets[user_idx],
                    'previous_reliability': previous_reliability,
                    'true_noise_rate_debug': true_noise_rate_debug,
                    'true_reliability_debug': true_reliability_debug,
                    'prediction_source': 'global',
                    **reliability_update
                }
            )
            print(
                '[Reliability Estimation] client={} n_confident={} '
                'agreement={} mean_confidence={} cycle_R={} R: {} -> {}'.format(
                    user_idx,
                    reliability_update['n_confident'],
                    reliability_update['agreement_rate'],
                    reliability_update['mean_confidence'],
                    reliability_update['cycle_reliability'],
                    previous_reliability,
                    reliability_update['updated_reliability']
                )
            )

        elif args.budget_mode in ['estimated_r_soft', 'estimated_r_soft_blend', 'shuffled_r_soft_blend']:
            observed_labels = np.asarray(
                [
                    _get_dataset_label(dataset_query, data_idx, args)
                    for data_idx in new_data
                ],
                dtype=np.int64
            )
            previous_reliability = reliability_state[user_idx]
            reliability_update = _update_soft_likelihood_reliability(
                probabilities,
                observed_labels,
                previous_reliability,
                args.num_classes,
                args.reliability_ema_beta
            )
            reliability_state[user_idx] = reliability_update['updated_reliability']

            # DEBUG/EVALUATION ONLY. NEVER USED BY THE ESTIMATOR OR BUDGET
            # ALLOCATION. Read only after the reliability state is updated.
            true_noise_rate_debug = _get_annotation_noise_rate(user_idx, args)
            true_reliability_debug = 1.0 - true_noise_rate_debug
            _write_soft_reliability_estimation_log(
                args.result_dir,
                args.current_ratio,
                {
                    'user_idx': user_idx,
                    'budget_mode': args.budget_mode,
                    'allocated_budget': budgets[user_idx],
                    'previous_reliability': previous_reliability,
                    'true_noise_rate_debug': true_noise_rate_debug,
                    'true_reliability_debug': true_reliability_debug,
                    'prediction_source': 'global',
                    **reliability_update
                }
            )
            print('[Soft Reliability Estimation]')
            print(
                'client={} n_query={} mean_p_observed={} mle_error={} '
                'cycle_R={} likelihood_range={} R: {} -> {}'.format(
                    user_idx,
                    reliability_update['n_query'],
                    reliability_update['mean_observed_probability'],
                    reliability_update['mle_error_rate'],
                    reliability_update['cycle_reliability'],
                    reliability_update['likelihood_range'],
                    previous_reliability,
                    reliability_update['updated_reliability']
                )
            )

        elif args.budget_mode == 'estimated_r_anchor_blend':
            # Query probabilities were computed before annotation; queried
            # labels are intentionally read only now, after annotation.
            observed_labels = np.asarray(
                [_get_dataset_label(dataset_query, data_idx, args) for data_idx in new_data],
                dtype=np.int64
            )
            previous_reliability = float(reliability_state[user_idx])
            query_estimate = _estimate_soft_likelihood_error(
                probabilities, observed_labels, args.num_classes
            )
            anchor_estimate = _estimate_soft_likelihood_error(
                anchor_probabilities, anchor_observed_labels, args.num_classes
            )

            denominator = float('nan')
            e_ann_raw = float('nan')
            e_ann_calibrated = float('nan')
            cycle_reliability = float('nan')
            calibration_boundary = 'none'
            valid = False
            skip_reason = ''
            if not query_estimate['valid']:
                skip_reason = 'query_{}'.format(query_estimate['skip_reason'])
            elif not anchor_estimate['valid']:
                skip_reason = 'anchor_{}'.format(anchor_estimate['skip_reason'])
            else:
                e_query = query_estimate['mle_error_rate']
                e_anchor = anchor_estimate['mle_error_rate']
                denominator = 1.0 - (
                    float(args.num_classes) / (float(args.num_classes) - 1.0)
                ) * e_anchor
                if denominator <= 1e-6:
                    skip_reason = 'nonpositive_calibration_denominator'
                else:
                    e_ann_raw = (e_query - e_anchor) / denominator
                    e_ann_calibrated = float(np.clip(e_ann_raw, 0.0, 1.0))
                    if e_ann_raw < 0.0:
                        calibration_boundary = 'lower'
                    elif e_ann_raw > 1.0:
                        calibration_boundary = 'upper'
                    cycle_reliability = 1.0 - e_ann_calibrated
                    reliability_state[user_idx] = float(np.clip(
                        float(args.reliability_ema_beta) * previous_reliability
                        + (1.0 - float(args.reliability_ema_beta)) * cycle_reliability,
                        0.0, 1.0
                    ))
                    valid = True

            # DEBUG/EVALUATION ONLY: this is deliberately read after all
            # estimator and reliability-state computations are complete.
            true_reliability_debug = 1.0 - _get_annotation_noise_rate(user_idx, args)
            anchor_true_reliability_debug[user_idx] = true_reliability_debug
            _write_anchor_reliability_estimation_log(
                args.result_dir,
                args.current_ratio,
                {
                    'user_idx': user_idx,
                    'budget_mode': args.budget_mode,
                    'allocated_budget': budgets[user_idx],
                    'n_query': query_estimate['n_samples'],
                    'n_anchor': anchor_estimate['n_samples'],
                    'e_query': query_estimate['mle_error_rate'],
                    'e_anchor': anchor_estimate['mle_error_rate'],
                    'denominator': denominator,
                    'e_ann_raw': e_ann_raw,
                    'e_ann_calibrated': e_ann_calibrated,
                    'query_likelihood_range': query_estimate['likelihood_range'],
                    'anchor_likelihood_range': anchor_estimate['likelihood_range'],
                    'query_mean_p_observed': query_estimate['mean_observed_probability'],
                    'anchor_mean_p_observed': anchor_estimate['mean_observed_probability'],
                    'query_mean_max_confidence': query_estimate['mean_max_confidence'],
                    'anchor_mean_max_confidence': anchor_estimate['mean_max_confidence'],
                    'query_boundary_flag': query_estimate['boundary_flag'],
                    'anchor_boundary_flag': anchor_estimate['boundary_flag'],
                    'calibration_boundary': calibration_boundary,
                    'cycle_reliability': cycle_reliability,
                    'previous_reliability': previous_reliability,
                    'updated_reliability': reliability_state[user_idx],
                    'valid': valid,
                    'skip_reason': skip_reason,
                    'true_reliability_debug': true_reliability_debug,
                    'prediction_source': 'global'
                }
            )
            print('[Anchor Soft Reliability Estimation] client={} e_query={} '
                  'e_anchor={} e_ann={} R: {} -> {} valid={} skip={}'.format(
                      user_idx, query_estimate['mle_error_rate'],
                      anchor_estimate['mle_error_rate'], e_ann_calibrated,
                      previous_reliability, reliability_state[user_idx], valid,
                      skip_reason or 'none'
                  ))

        # Labels actually observed by the client AFTER annotation.
        count_query_observed = get_label_count(
            dataset_query,
            new_data,
            args
        )

        if count_query_true is not None:
            print(
                "count_query_true: ",
                count_query_true
            )

        print(
            "count_query_observed: ",
            count_query_observed
        )
        time += datetime.datetime.now() - start
        
        print(args.al_method, user_idx)
        print("(Before) Label examples: {}".format(len(label_idxs)))
        required_budget = (
            int(args.n_query / args.num_users)
            if args.budget_mode == 'fixed' else budgets[user_idx]
        )
        if len(new_data) < required_budget:
            sys.exit("too few remaining examples to query")

        dict_users_train_label[user_idx] = np.array(list(new_data) + list(label_idxs))   
        print("(After) Label examples: {}".format(len(list(new_data)) + len(label_idxs))) 
        
        with open(results_save_path, "a") as f:
            f.write(f"User {user_idx}\n")
            if count_unlabel is not None:
                f.write(f"count_unlabel: {count_unlabel}\n")
            if count_query_true is not None:
                f.write(
                    f"count_query_true: {count_query_true}\n"
                )

            f.write(
                f"count_query_observed: "
                f"{count_query_observed}\n"
            )

            f.write(
                f"annotation_noise: "
                f"{noise_stats['n_corrupt']}/"
                f"{noise_stats['n_query']}\n"
            )
            f.write(f"global model test accuracy: {global_acc}, local model test accuracy: {local_acc}\n\n")

    time /= len(dict_users_train_total)     
    print('Querying instances takes {}'.format(time))           

    if args.budget_mode == 'estimated_r_anchor_blend':
        _print_anchor_reliability_diagnostic(
            reliability_state, anchor_true_reliability_debug
        )

    # Save dict_users for next round
    path = os.path.join(args.dict_user_path, 'dict_users_train_label_{:.3f}.pkl'.format(args.current_ratio))
    with open(path, 'wb') as handle:
        pickle.dump(dict_users_train_label, handle)

    return dict_users_train_label, mean_global_ratio
    
