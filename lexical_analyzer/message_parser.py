import os
from typing import Dict, Union, List

import nltk

from lexical_analyzer import vocabulary_size, UNKNOWN_TOKEN


class Message:
    def __init__(self, subject: str, msg: str, is_spam: bool):
        self.msg_subject = subject
        self.msg_body = msg
        self.is_spam = is_spam


class MessageParser:
    def __init__(self, path: str, spam_msg_limit: int = 200, clear_msg_limit: int = 200):
        self.messages = []
        self.proceeded_spam_msgs = 0
        self.proceeded_clear_msgs = 0
        self.__pars_directory_files(path, spam_msg_limit, clear_msg_limit)
        self.max_message_len = 0
        self.words_index = MessageParser.__words_indexer(self.__generate_common_dictionary())

    def __pars_directory_files(self, path: str, spam_msg_limit: int, clear_msg_limit: int):
        for dir_path, dir_names, file_names, in os.walk(path):
            for filename in file_names:
                if self.proceeded_spam_msgs < spam_msg_limit and 'spm' in filename:
                    self.__parse_file(dir_path, filename, True)
                    self.proceeded_spam_msgs += 1
                elif self.proceeded_clear_msgs < clear_msg_limit and 'spm' not in filename:
                    self.__parse_file(dir_path, filename, False)
                    self.proceeded_clear_msgs += 1
        print('All - {} / Spam - {} / Clear = {} messages proceed'
              .format(len(self.messages), self.proceeded_spam_msgs, self.proceeded_clear_msgs))

    def __parse_file(self, dir_path: str, filename: str, is_spam: bool):
        with open(dir_path + '/' + filename, 'r') as file:
            text = file.read()
            split_list = text.split(sep='\n')
            if len(split_list) != 4:
                print('len(split_list) != 4, parsing failed')
                return
            message = Message(split_list[0].lower(), split_list[2].lower(), is_spam)
            self.messages.append(message)

    def __generate_common_dictionary(self, most_common: int = vocabulary_size, with_frequency: bool = False):
        self.all_messages_in_str = ''.join(message.msg_body for message in self.messages)
        messages_tokens = nltk.word_tokenize(self.all_messages_in_str)
        common_words_with_freq = nltk.FreqDist(messages_tokens).most_common(most_common)
        return common_words_with_freq if with_frequency else [word[0] for word in common_words_with_freq]

    @staticmethod
    def __words_indexer(common_tokenized_vocabulary: List[str], reverse: bool=False) -> Dict[str, int]:
        if reverse:
            return {index: word for index, word in enumerate(common_tokenized_vocabulary, start=UNKNOWN_TOKEN + 1)}
        else:
            return {word: index for index, word in enumerate(common_tokenized_vocabulary, start=UNKNOWN_TOKEN + 1)}

    def index_by_message(self) -> Union[List[List[int]], List[int]]:
        indexed_messages = []
        target_verdicts = []
        for message in self.messages:
            target_verdicts.append(int(message.is_spam))
            tokenized_message = nltk.word_tokenize(message.msg_body)
            indexed_messages.append([self.words_index.get(word, UNKNOWN_TOKEN) for word in tokenized_message])
            self.max_message_len = max(self.max_message_len, len(tokenized_message))
        return indexed_messages, target_verdicts
