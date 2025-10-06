import pickle
import numpy as np
from collections import defaultdict
from uncertainty.utils import openai as oai
from tqdm import tqdm


def gpt_metric(predicted_answer, example, model, ask_ref=True, temperature=0.01):
    if 'answers' in example:
        correct_answers = example['answers']['text']
    elif 'reference' in example:
        correct_answers = example['reference']['answers']['text']
    elif 'answer' in example:
        correct_answers = example['answer']
    else:
        raise ValueError

    prompt = f'We are assessing the quality of answers to the following question: {example["question"]}\n'
    if ask_ref:
        if len(correct_answers) == 1:
            prompt += f"The expected answer is: {correct_answers[0]}.\n"
        else:
            prompt += f"The following are expected answers to this question: {correct_answers}.\n"

    prompt += f"The proposed answer is: {predicted_answer}\n"

    if len(correct_answers) == 1:
        if ask_ref:
            prompt += "On the basis of the given question, expected answer, context and your own knowledge, is the proposed answer correct? Please think carefully and"
        else:
            prompt += "Based on the context of question and your own knowledge, is the proposed answer correct? Please think carefully and"
    else:
        prompt += "Within the context of the question and your own knowledge, does the proposed answer mean the same as any of the expected answers?"

    prompt += " Respond only with yes or no.\nResponse:"
    # print(prompt)

    predicted_answer = oai.predict(prompt, temperature, model = model)

    if 'yes' in predicted_answer.lower():
        return 1.0
    elif 'no' in predicted_answer.lower():
        return 0.0
    else:
        print('Redo llm check.')
        predicted_answer = oai.predict(prompt, 1, model = model)
        if 'yes' in predicted_answer.lower():
            return 1.0
        elif 'no' in predicted_answer.lower():
            return 0.0

        print('Answer neither no nor yes. Defaulting to no!')
        return 0.0
    

def main(args, dataset_split):
    filename = f"{args.data_dir}/qa_{dataset_split}_{args.data_name}_{args.model_name}.pkl"
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(f"Data: {filename}, {len(data)}")

    args.use_num_generations = 1 if dataset_split=='train' else 5
    
    accuracy_dict = defaultdict(list)
    for i, tid in tqdm(enumerate(data)):
        example = data[tid]
        question = example['question']
        answer = example['answer']
        print(f"{i} == Question: {question}, \nGold Answer: {answer}")
        predicted_answer = example['high_t_answer']
        print(len(predicted_answer))

        for j, pred_ in enumerate(predicted_answer):
            if j == args.use_num_generations:
                break
            acc = gpt_metric(pred_[0], example, args.eval_model)
            accuracy_dict[tid].append(acc)
        #     print(f"Predicted Answer: {pred_[0]}")
        # print(f"Accuracy: {accuracy_dict[tid]}")

        # Save the accuracy_dict to a file
        if (i+1) % 100 == 0:
            acc_filename = f"{args.data_dir}/acc_{dataset_split}_{args.data_name}_{args.model_name}.pkl"
            with open(acc_filename, 'wb') as f:
                pickle.dump(accuracy_dict, f)
    
    acc_all = np.array([np.array(v) for v in accuracy_dict.values()])
    print(f"Accuracy: {acc_all.shape}, {np.sum(acc_all, axis=0)}")

    # double check the accuracy_dict 
    if args.double_check:
        accuracy_dict_double = defaultdict(list)
        for i, tid in enumerate(data):
            example = data[tid]
            question = example['question']
            answer = example['answer']
            predicted_answer = example['high_t_answer']

            acc_list = accuracy_dict[tid]
            for j, pred_ in enumerate(predicted_answer):
                if j == args.use_num_generations:
                    break
                acc = acc_list[j]
                if acc == 0.0:
                    acc = gpt_metric(pred_[0], example, args.eval_model, ask_ref=False)
                    if acc == 1.0:
                        acc = 2.0
                    # print(f"{i}, Double check: {question}, \n{answer}, \nacc: === {acc}, {pred_[0]}")
                accuracy_dict_double[tid].append(acc)

            if (i+1) % 100 == 0:
                acc_filename = f"{args.data_dir}/acc_double_{dataset_split}_{args.data_name}_{args.model_name}.pkl"
                with open(acc_filename, 'wb') as f:
                    pickle.dump(accuracy_dict_double, f)
        
        acc_all = np.array([np.array(v) for v in accuracy_dict_double.values()])
        print(f"Accuracy: {acc_all.shape}, {np.sum(acc_all, axis=0)}")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='your data directory')
    parser.add_argument('--data_name', type=str, default='squad', help='trivia_qa, squad, nq, bioasq')
    parser.add_argument('--model_name', type=str, default='llama3_70b')
    parser.add_argument('--use_num_generations', type=int, default=6)
    parser.add_argument('--eval_model', default='gpt-4.1', type=str)
    parser.add_argument("--entailment_model", default='gpt-3.5', type=str)
    parser.add_argument('--double_check', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--strict_entailment',
                        default=True, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    for dataset_split in ['train', 'validation']:
        # for args.data_name in ['trivia_qa', 'squad', 'nq', 'bioasq']:
        #     print(f"Evaluating {dataset_split} data: {args.data_name}")
        main(args, dataset_split)


    


