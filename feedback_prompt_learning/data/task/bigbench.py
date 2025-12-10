# define task prompts for various datasets
import re
import string

from .base_task import BaseDataset, BaseTask

number_to_word_dict = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "twenty-one": 21
}


class CustomTask(BaseTask):
    def __init__(self,
                 train_size,
                 eval_size,
                 test_size=None,

                 task_name="bigbench",
                 task_description="task from bigbench",
                 data_dir='',
                 seed=None,

                 post_instruction=True,
                 TaskDataset=BaseDataset,
                 option_num=5,
                 **kwargs):
        self.options = {}
        super().__init__(
            task_name=task_name,
            task_description=task_description,
            data_dir=data_dir,
            seed=seed,
            train_size=train_size,
            eval_size=eval_size,
            test_size=test_size,
            post_instruction=post_instruction,
            TaskDataset=TaskDataset,
            option_num=option_num,
        )

        # Task-specific configurations
        if task_name == "object_counting":
            self.answer_format_prompt = ""
        elif task_name == "epistemic":
            self.answer_format_prompt = "\nA:"

    def load_task_dataset(self, data_dir):
        '''
            <task specific>
        '''
        json_data = self._load_json_file(data_dir)
        self.task_description = json_data['description']

        # Only for bigbench task (not object_counting or epistemic)
        if self.task_name not in ["object_counting", "epistemic"]:
            max_example = max(json_data['examples'], key=lambda x: len(x['target_scores']))
            self.option_num = len(max_example['target_scores'])

        return json_data

    def transform_format(self, data):
        if self.task_name == "object_counting":
            return self._transform_object_counting(data)
        elif self.task_name == "epistemic":
            return self._transform_epistemic(data)
        else:
            return self._transform_bigbench(data)

    def _transform_bigbench(self, data):
        original_examples = data['examples']
        examples = []

        for example in original_examples:
            question = example['input']
            if 'task_prefix' in data.keys():
                task_prefix = data['task_prefix'].strip()
                question = task_prefix + "\n" + question

            target_scores = example['target_scores']

            # Generating options and answer
            options = list(target_scores.keys())
            answer = [chr(65 + i) for i, option in enumerate(options) if target_scores[option] == 1][0]
            for i, option in enumerate(options):
                self.options[option.lower()] = f'{chr(65 + i)}'
            options = [f'({chr(65 + i)}) {option}' for i, option in enumerate(options)]
            options_str = 'Options:\n' + '\n'.join(options)
            question_str = question + '\n' + options_str + '\n'

            formatted_example = {
                'question': question_str,
                'answer': answer
            }
            examples.append(formatted_example)

        return examples

    def _transform_object_counting(self, data):
        original_examples = data['examples']
        examples = []

        for example in original_examples:
            question = example['input']
            answer = example['target']
            formatted_example = {
                'question': question,
                'answer': answer[-1]
            }
            examples.append(formatted_example)

        return examples

    def _transform_epistemic(self, data):
        original_examples = data['examples']
        examples = []

        for example in original_examples:
            question = example['input']
            target_scores = example['target_scores']

            # Generating options and answer
            options = list(target_scores.keys())
            answer = [option.lower() for i, option in enumerate(options) if target_scores[option] == 1][0]

            options_str = 'Options:\n- entailment\n- non-entailment'
            question_str = "Identify the relation between the following premises and hypotheses, choosing from the options 'entailment' or 'non-entailment'.\n" + question + "\n" + options_str + '\n'

            formatted_example = {
                'question': question_str,
                'answer': answer
            }
            examples.append(formatted_example)

        return examples

    def clean_response(self, response):
        if self.task_name == "object_counting":
            return self._clean_object_counting(response)
        elif self.task_name == "epistemic":
            return self._clean_epistemic(response)
        else:
            return self._clean_bigbench(response)

    def _clean_bigbench(self, response):
        letters = string.ascii_uppercase[:self.option_num] + string.ascii_lowercase[:self.option_num]
        clean_pattern = r"<answer>([\s\S]*?)<\/answer>"
        match = re.findall(clean_pattern, response.lower())

        if len(match) == 0 or not match[-1].strip():
            pattern_str = '|'.join([re.escape(option) for option in self.options])
            backup_match = re.findall(pattern_str, response, re.IGNORECASE)

            if backup_match:
                return self.options[backup_match[-1].lower()]
            else:
                return 'N/A: Format error'

        answer = re.search(r"\([" + letters + r"]\)", match[-1])
        if answer is not None:
            return answer.group(0)[1].upper()
        answer = re.search(r"[" + letters + r"]", match[-1])
        if answer is None:
            return 'N/A: Format error'
        return answer[0].upper()

    def _clean_object_counting(self, response):
        integer_pattern = r"\d+"
        matches = re.findall(integer_pattern, response)
        if len(matches) != 0:
            return str(matches[-1])

        extended_pattern = r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty-one)\b"
        matches = re.findall(extended_pattern, response)
        if len(matches) != 0:
            return str(number_to_word_dict[matches[-1]])
        else:
            return "N/A: format error."

    def _clean_epistemic(self, response):
        clean_pattern = r"\b(entailment|non-entailment)\b"
        match = re.findall(clean_pattern, response.lower())
        if len(match) != 0:
            return match[-1]

        return "N/A: format error."
