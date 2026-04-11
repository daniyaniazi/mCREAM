from ctypes import c_double
from src.models import Template_CBM_MultiClass, UtoY_model
from typing import Any, Optional, Union, Callable
import torch
from torch import Tensor,BoolTensor
import random


def invert_softmax(index_of_desired_logit, logit_vector_mutex_group, concept_activation):
    eps = torch.finfo(concept_activation.dtype).eps
    # Clamp activation to (0, 1)
    # concept_activation = torch.clamp(concept_activation, min=eps, max=1 - eps)
    concept_activation = torch.clamp(concept_activation, min=eps, max=1-eps)

    # Remove the desired logit
    logits_wo_desired = torch.cat(
        [
            logit_vector_mutex_group[:, :index_of_desired_logit],
            logit_vector_mutex_group[:, index_of_desired_logit + 1 :],
        ],
        dim=1,
    )

    # log(sum_{j ≠ i} exp(logit_j))
    log_sum_without_desired = torch.logsumexp(logits_wo_desired, dim=1)

    intervened_logits = (
        torch.log(concept_activation / (1 - concept_activation))
        + log_sum_without_desired
    )

    return intervened_logits

def clamped_invert_softmax(index_of_desired_logit, logit_vector_mutex_group, concept_activation):
    """Clamping improves the results of just inverting."""
    intervened_logits = invert_softmax(index_of_desired_logit, logit_vector_mutex_group, concept_activation)
    
    ### clamping, used for ReLU

    intervened_logits = torch.clamp(concept_activation, min=0)

    return intervened_logits




class UtoY_model_propagating_interventions(UtoY_model):
    def __init__(self, 
                 num_exogenous: int, 
                 num_concepts: int, 
                 num_side_channel: int, 
                 num_classes: int, 
                 learning_rate: float = 0, 
                 lambda_weight: Any = None, 
                 causal_graph: BoolTensor | None = None, 
                 masking_algorithm: str = "zuko", 
                 num_hidden_layers_in_maskedmlp: int = 0, 
                 previous_model_output_size: int | None = None, 
                 last_layer_mask: bool = False, 
                 concept_representation: str = "logit", 
                 side_dropout: bool = False, 
                 dropout_prob: float = 0, 
                 mutually_exclusive_concepts: list | None = None, 
                 **kwargs: Any) -> None:
        
        super().__init__(num_exogenous, 
                         num_concepts, 
                         num_side_channel, 
                         num_classes, 
                         learning_rate, 
                         lambda_weight, 
                         causal_graph, 
                         masking_algorithm, 
                         num_hidden_layers_in_maskedmlp, 
                         previous_model_output_size, 
                         last_layer_mask, 
                         concept_representation, 
                         side_dropout, 
                         dropout_prob, 
                         mutually_exclusive_concepts, 
                         **kwargs)

        self.maskedmlp_depth=num_hidden_layers_in_maskedmlp
    
    def forward_with_interventions(
        self,
        x: Tensor,
        true_concepts: Tensor,
        num_interventions: int = 1,
        intervention_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Based on inversion of softmax"""
        input_per_concept = (self.num_exogenous - self.num_side_channel) // self.num_concepts
        assert input_per_concept == 1
        assert self.maskedmlp_depth==0 ## for higher depth: need invertible activation functions



        ### get the weights and biases, (and mask) for inversion
        last_layer = self.u2c_model[-1]
        with torch.no_grad():
            import pdb
            pdb.set_trace()
            # masks
            mask0 = last_layer.mask[0]   # BoolTensor [in_features]
            mask5 = last_layer.mask[5]

            # raw (unmasked) weights
            w0_raw = last_layer.weight[0]
            w5_raw = last_layer.weight[5]

            # select only active connections
            w0_active = w0_raw[mask0]
            w5_active = w5_raw[mask5]

            b0 = last_layer.bias[0]
            b5 = last_layer.bias[5]

        ### assert if last mask is invertible 

        u = self.u2u_model(x)

        # [Uc;Uy]
        Uc = u[:, : self.num_exogenous - self.num_side_channel]
        Uy = u[:, self.num_exogenous - self.num_side_channel :]

        
        # clone original exogenous variables
        Uc_intervened = Uc.clone()


        # import pdb
        # pdb.set_trace()
        c = self.u2c_model(Uc)

        logits_before_softmax = c.clone()

        c = self.concept_activation_function(c)
        c_predicted = c.clone()
        ### create the intervened concept vector (before propagation)
        if intervention_mask is None:  # if its not given by another method
            if self.group_interventions:  
                raise NotImplementedError("propagating interventions implemented only for individual interventions")
            else:
                intervention_mask = self.generate_intervention_mask(
                    num_interventions=num_interventions,
                    batch_size=c.size(0)
                )
        
        # concepts after interventions, no propagation
        c_predicted[intervention_mask] = (true_concepts[intervention_mask]).type(
            c_predicted.dtype
        )

        if num_interventions<1:
            # get batch and concept indices where intervention_mask is True
            batch_idxs, concept_idxs = intervention_mask.nonzero(as_tuple=True)

            for b_idx, c_idx in zip(batch_idxs, concept_idxs):
                if c_idx not in (0,5):
                    #### ignore indirect concepts
                    continue
                # find the mutex group that contains this concept
                mutex_group_indices = None
                for group in self.mutually_exclusive_concepts:
                    if c_idx.item() in group:
                        mutex_group_indices = group
                        break
                if mutex_group_indices is None:
                    raise ValueError(f"Concept index {c_idx.item()} not found in any mutex group.")

                # extract the logits for this mutex group
                mutex_group_logits = logits_before_softmax[b_idx, mutex_group_indices]

                # compute the intervened logit for this concept
                intervened_logit = clamped_invert_softmax(
                    index_of_desired_logit=mutex_group_indices.index(c_idx.item()),
                    logit_vector_mutex_group=mutex_group_logits.unsqueeze(0),  # add batch dim
                    concept_activation=true_concepts[b_idx, c_idx].unsqueeze(0)
                )


                # write it back into Uc_intervened
                if c_idx==0:
                    weight= w0_active
                    bias= b0
                elif c_idx==5:
                    weight= w5_active
                    bias= b5
                
                Uc_intervened[b_idx, c_idx] = torch.clamp((intervened_logit -bias)/weight, min=0)
                # propagate interventions
                logits_before_softmax=self.u2c_model(Uc_intervened) # now forward pass the new exogenous variables
                

            ##### add again the intervened values to the concept vector (fail-safe code)
            c = self.concept_activation_function(logits_before_softmax)
            c_predicted = c.clone()
            c_predicted[intervention_mask] = (true_concepts[intervention_mask]).type(
                c_predicted.dtype
            )
        c = c_predicted  # CHANGE the new C TO THE TRUE INTERVENED VALUES
        #####

        # continue with forward pass
        # CBM + side channel
        if self.side_dropout is True and self.masking_algorithm == "none":
            s = self.side_channel(Uc) #yes its Uc
            last_layer_input = torch.cat((c, s), dim=1)
            y = self.last_layer(last_layer_input)
        # CREAM + side channel
        elif self.num_side_channel > 0:
            s = self.side_channel(Uy)
            last_layer_input = torch.cat((c, s), dim=1)
            y = self.last_layer(last_layer_input)

        else:
            y = self.last_layer(c)
        return y, c


    def forward_with_interventions_swapping(
        self,
        x: Tensor,
        true_concepts: Tensor,
        num_interventions: int = 1,
        intervention_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        """Current implementation should only work for the FMNIST datasets, that use softmax activations. 
        For an alternative more general implementation look at: 
        https://github.com/adrianjav/causal-flows/blob/282f245be3228cadde53fba4b74e9aac1600f635/causalflows/distributions.py#L17"""
        input_per_concept = (self.num_exogenous - self.num_side_channel) // self.num_concepts
        assert input_per_concept == 1
        assert self.maskedmlp_depth==0 ## for higher depth: need invertible activation functions

        u = self.u2u_model(x)

        # [Uc;Uy]
        Uc = u[:, : self.num_exogenous - self.num_side_channel]
        Uy = u[:, self.num_exogenous - self.num_side_channel :]


        # import pdb
        # pdb.set_trace()
        c = self.u2c_model(Uc)
        c = self.concept_activation_function(c)
        c_predicted = c.clone()
        ### create the intervened concept vector (before propagation)
        if intervention_mask is None:  # if its not given by another method
            if self.group_interventions:  
                raise NotImplementedError("propagating interventions works only for individual interventions")
            else:
                intervention_mask = self.generate_intervention_mask(
                    num_interventions=num_interventions,
                    batch_size=c.size(0)
                )
        
        c_predicted[intervention_mask] = (true_concepts[intervention_mask]).type(
            c_predicted.dtype
        )

        # clone original exogenous variables
        Uc_intervened = Uc.clone()

        # get batch and concept indices where intervention_mask is True
        batch_idxs, concept_idxs = intervention_mask.nonzero(as_tuple=True)

        for b_idx, c_idx in zip(batch_idxs, concept_idxs):
            # find the mutex group that contains this concept
            mutex_group_indices = None
            for group in self.mutually_exclusive_concepts:
                if c_idx.item() in group:
                    mutex_group_indices = group
                    break

            if mutex_group_indices is None:
                raise ValueError(f"Concept index {c_idx.item()} not found in any mutex group.")

            group_size = len(mutex_group_indices)

            if group_size == 2:
                import pdb
                pdb.set_trace()
                i, j = mutex_group_indices
                # determine which index is the one being intervened
                if c_idx.item() == i:
                    other_idx = j
                else:
                    other_idx = i

                # directional intervention based on true_concepts
                if true_concepts[b_idx, c_idx] > true_concepts[b_idx, other_idx]:
                    Uc_intervened[b_idx, c_idx] = torch.max(
                        Uc_intervened[b_idx, c_idx], Uc_intervened[b_idx, other_idx]
                    )
                else:
                    Uc_intervened[b_idx, c_idx] = torch.min(
                        Uc_intervened[b_idx, c_idx], Uc_intervened[b_idx, other_idx]
                    )

            else:
                # fallback for larger mutex groups: just copy original logit
                Uc_intervened[b_idx, c_idx] = Uc[b_idx, c_idx]



        # propagate interventions
        c = self.u2c_model(Uc_intervened) # now forward pass the new exogenous variables
        c = self.concept_activation_function(c)

        ##### add again the intervened values to the concept vector (fail-safe code)
        c_predicted = c.clone()
        c_predicted[intervention_mask] = (true_concepts[intervention_mask]).type(
            c_predicted.dtype
        )
        c = c_predicted  # CHANGE the new C TO THE TRUE INTERVENED VALUES
        #####

        # continue with forward pass
        # CBM + side channel
        if self.side_dropout is True and self.masking_algorithm == "none":
            s = self.side_channel(Uc) #yes its Uc
            last_layer_input = torch.cat((c, s), dim=1)
            y = self.last_layer(last_layer_input)
        # CREAM + side channel
        elif self.num_side_channel > 0:
            s = self.side_channel(Uy)
            last_layer_input = torch.cat((c, s), dim=1)
            y = self.last_layer(last_layer_input)

        else:
            y = self.last_layer(c)
        return y, c    

    def generate_intervention_mask(
        self, num_interventions: int, batch_size: int ) -> Tensor:
        """Same function as original class, but now will prioritize indirect concepts."""
        indirect_concept_indices, direct_concept_indices = (
            self._split_into_in_direct_concepts()
        )

        all_intervention_masks = []

        if num_interventions > self.num_concepts:
            raise ValueError("Num of interventions exceeds number of concepts")

        for _ in range(batch_size):
            intervention_mask = torch.zeros(self.num_concepts, dtype=torch.bool)
            selected_indices = random.sample(
                indirect_concept_indices,
                min(num_interventions, len(indirect_concept_indices)),
            )

            intervention_mask[selected_indices] = 1

            if num_interventions > len(indirect_concept_indices):
                selected_indices = random.sample(
                    direct_concept_indices,
                    num_interventions - len(indirect_concept_indices),
                )
                intervention_mask[selected_indices] = 1

            all_intervention_masks.append(intervention_mask)

        return torch.stack(all_intervention_masks)
    