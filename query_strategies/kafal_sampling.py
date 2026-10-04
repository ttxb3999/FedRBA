import os
import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .strategy import Strategy, DatasetSplit


class KAFALSampling(Strategy):
    """
    KAFAL KSAS query strategy adapted to the FairFAL framework.

    Global model:
        self.net

    Local model:
        final-round client model saved during the previous
        active-learning stage.
    """

    def __init__(self, dataset_query, dataset_train,
                 dataset_test, net, args):
        super().__init__(
            dataset_query,
            dataset_train,
            dataset_test,
            net,
            args
        )

    def query_model_selection(
        self,
        user_idx,
        label_idxs,
        unlabel_idxs,
        count_label
    ):
        # FairFAL's interface expects this function.
        # KAFAL does not use FairFAL's global/local selector score.
        return 0.0

    def _load_local_model(self, user_idx):
        # args.query_model is the previous AL stage's last.pt.
        previous_result_dir = os.path.dirname(self.args.query_model)

        local_model_path = os.path.join(
            previous_result_dir,
            'local_models',
            'user_{}.pt'.format(user_idx)
        )

        if not os.path.exists(local_model_path):
            raise FileNotFoundError(
                'KAFAL local model not found: {}'.format(
                    local_model_path
                )
            )

        local_net = copy.deepcopy(self.net)

        state_dict = torch.load(
            local_model_path,
            map_location=self.args.device
        )

        local_net.load_state_dict(state_dict)
        local_net.to(self.args.device)
        local_net.eval()

        return local_net

    def _get_discrepancy(
        self,
        local_net,
        unlabel_idxs,
        count_label
    ):
        """
        Knowledge-Specialized Active Sampling (KSAS).

        Based on the original KAFAL implementation:
        1. add client class-count log prior to both logits;
        2. compute symmetric KL divergence between
           local and global predictions.
        """

        loader = DataLoader(
            DatasetSplit(self.dataset_query, unlabel_idxs),
            batch_size=self.args.test_bs,
            shuffle=False
        )

        num_classes = self.args.num_classes
        device = self.args.device

        # Same meaning as loss_weight_list[c]
        # in the original KAFAL implementation.
        spc = torch.zeros(
            num_classes,
            dtype=torch.float32,
            device=device
        )

        for cls, cnt in count_label.items():
            spc[int(cls)] = float(cnt)

        # Original KAFAL uses spc.log()
        log_spc = torch.log(spc + 1e-6)

        discrepancy = torch.zeros(
            len(unlabel_idxs),
            dtype=torch.float32
        )

        local_net.eval()
        self.net.eval()

        with torch.no_grad():
            for inputs, labels, positions in loader:
                inputs = inputs.to(device)

                local_scores, _ = local_net(inputs)
                global_scores, _ = self.net(inputs)

                # Intensify client-specialized knowledge.
                local_scores = local_scores + log_spc.unsqueeze(0)
                global_scores = global_scores + log_spc.unsqueeze(0)

                kl_local_global = F.kl_div(
                    F.log_softmax(local_scores, dim=1),
                    F.softmax(global_scores, dim=1),
                    reduction='none'
                ).sum(dim=1)

                kl_global_local = F.kl_div(
                    F.log_softmax(global_scores, dim=1),
                    F.softmax(local_scores, dim=1),
                    reduction='none'
                ).sum(dim=1)

                score = 0.5 * (
                    kl_local_global + kl_global_local
                )

                discrepancy[positions] = score.cpu()

        if torch.isnan(discrepancy).any():
            raise RuntimeError(
                'NaN appeared in KAFAL discrepancy scores.'
            )

        return discrepancy

    def query(
        self,
        user_idx,
        label_idxs,
        unlabel_idxs,
        test_idxs,
        count_label,
        mean_global_ratio,
        n_query
    ):
        local_net = self._load_local_model(user_idx)

        # Original KAFAL shuffles the unlabeled pool
        # before ranking. This also gives random tie-breaking.
        candidate_idxs = list(unlabel_idxs)
        random.shuffle(candidate_idxs)

        discrepancy = self._get_discrepancy(
            local_net,
            candidate_idxs,
            count_label
        )

        n_query = min(n_query, len(candidate_idxs))

        selected_positions = torch.topk(
            discrepancy,
            k=n_query,
            largest=True
        ).indices.tolist()

        new_data = [
            candidate_idxs[pos]
            for pos in selected_positions
        ]

        global_acc = self.test(
            self.net,
            test_idxs
        )

        local_acc = self.test(
            local_net,
            test_idxs
        )

        print(
            'KAFAL user {} discrepancy '
            'min={:.6f}, mean={:.6f}, max={:.6f}'.format(
                user_idx,
                discrepancy.min().item(),
                discrepancy.mean().item(),
                discrepancy.max().item()
            )
        )

        del local_net

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return new_data, global_acc, local_acc