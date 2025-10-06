"""Sample answers from LLMs on QA task."""
import gc
import os
import random
from tqdm import tqdm
import pickle

import numpy as np
import torch

from uncertainty.data.data_utils import load_ds
from uncertainty.utils import utils


def main(args):

    # Setup run.
    if args.data_name == 'squad':
        if not args.answerable_only:
            print('Forcing `answerable_only=True` for squad dataset.')
            args.answerable_only = True
    train_dataset, validation_dataset = load_ds(args.data_name, seed=args.random_seed)

    # Get indices of answerable and unanswerable questions and construct prompt.
    answerable_indices, unanswerable_indices = utils.split_dataset(train_dataset)
    del unanswerable_indices
    train_dataset = [train_dataset[i] for i in answerable_indices]

    val_answerable, val_unanswerable = utils.split_dataset(validation_dataset)
    del val_unanswerable
    validation_dataset = [validation_dataset[i] for i in val_answerable]

    # basic prompts
    BRIEF = 'Answer the following question in a single but complete sentence only.\n'

    # Initialize model.
    model = utils.init_model(args)

    # Start generation.
    for dataset_split in ['train', 'validation']:
        print(60 * '=', f'starting generation with {dataset_split}', 60 * '=')

        if dataset_split == 'train':
            if not args.get_training_set_generations:
                print('Skip training data.')
                continue
            dataset = train_dataset
            possible_indices = range(0, len(dataset))
            print('Train dataset:', len(possible_indices))

        else:
            dataset = validation_dataset
            possible_indices = range(0, len(dataset))
            print('Validation dataset:', len(possible_indices))

        # Evaluate over random subset of the datasets.
        random.seed(args.random_seed)
        num_samples = args.train_num_samples if dataset_split == 'train' else args.valid_num_samples
        indices = random.sample(possible_indices, min(num_samples, len(dataset)))
        print(f'{dataset_split}_Indices: {len(indices)}')

        # This will store all input data and model predictions.
        generations = {}
        qa_generations = {}
        it = 0
        for index in tqdm(indices):
            # Grab example at index.
            example = dataset[index]
            question, context = example["question"], example['context']
            correct_answer = example['answers']['text']

            generations[example['id']] = {'question': question, 'context': context}
            generations[example['id']]['answer'] = correct_answer
            qa_generations[example['id']] = {'question': question, 'context': context, 'answer': correct_answer}

            # Construct prompt with input.
            if args.use_context:
                lc = min(len(context), 1500)
                prompt = BRIEF + f"Passage: {context[:lc]}\n" + f"Question: {question}\n" + "Answer:"
            else:
                prompt = BRIEF + f"Question: {question}\n" +"Answer:"
            print(f"local_prompt: {prompt}")

            # sampling generations
            full_responses = []
            qa_responses = []
            for i in range(args.num_generations):
                temperature = args.temperature
                print(f"n_generation{i}, temperature: {temperature}")
                
                if i == 0:
                    predicted_answer, predicted_tokens, token_log_likelihoods, token_entropy, hidden_input, hidden_output = model.predict(
                        prompt, temperature, args.return_layers)
                    while predicted_answer == 0:
                        predicted_answer, predicted_tokens, token_log_likelihoods, token_entropy, hidden_input, hidden_output = model.predict(
                            prompt, temperature, args.return_layers)
                    hidden_input = hidden_input.cpu()
                else:
                    hidden_input = None
                    predicted_answer, predicted_tokens, token_log_likelihoods, token_entropy, _, hidden_output = model.predict(
                        prompt, temperature, args.return_layers)
                    while predicted_answer == 0:
                        predicted_answer, predicted_tokens, token_log_likelihoods, token_entropy, _, hidden_output = model.predict(
                            prompt, temperature, args.return_layers)

                hidden_output = hidden_output.cpu() if hidden_output is not None else None # (n_tokens, n_layers, hidden_dim)
                predicted_tokens = predicted_tokens.cpu()

                print(f"predicted_answer: {predicted_answer}")

                # Aggregate predictions over num_generations.
                if i > 0 and dataset_split=='train':
                    full_responses.append(
                        (predicted_answer, predicted_tokens, token_log_likelihoods, token_entropy.tolist()))
                else:
                    full_responses.append(
                        (predicted_answer, predicted_tokens, token_log_likelihoods, token_entropy.tolist(), hidden_input, hidden_output))
                qa_responses.append([predicted_answer, token_log_likelihoods])

            # Append all predictions for this example to `generations`.
            generations[example['id']]['generated_answer'] = full_responses
            qa_generations[example['id']]['generated_answer'] = qa_responses

            # changes required for saving data
            if (it+1) % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                if not os.path.exists(args.data_dir):
                    os.makedirs(args.data_dir  )
                with open(f"{args.data_dir}/{dataset_split}_{args.data_name}_{args.model_name}.pkl", 'wb') as file:
                    pickle.dump(generations, file)
                with open(f"{args.data_dir}/qa_{dataset_split}_{args.data_name}_{args.model_name}.pkl", 'wb') as file:
                    pickle.dump(qa_generations, file)
        
            it += 1
            print('Finished with example:', it)

    print('Run complete.')
    del model



if __name__ == '__main__':
    parser = utils.get_parser()
    args, unknown = parser.parse_known_args()

    # First sample generations from LLM.
    main(args)