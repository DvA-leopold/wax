import os


class Message:
    def __init__(self, subject, msg, is_spam=False):
        self.is_spam = is_spam
        self.msg_subject = subject
        self.msg_body = msg


class MessageParser:
    def __init__(self):
        self._messages = []

    def init_sub_files(self, path):
        for dir_path, dir_names, file_names, in os.walk(path):
            for filename in file_names:
                try:
                    self._parse_file(dir_path, filename)
                except IndexError as err:
                    print('Index except: ', err.args)
        print('Proceed %d messages.' % len(self._messages))

    def _parse_file(self, dir_path, filename):
        file = open(dir_path + '/' + filename, 'r')
        text = file.read()
        split_list = text.split(sep='\n')
        if len(split_list) != 4:
            file.close()
            raise IndexError('split list wrong size:', len(split_list), ' !')
        is_spam = 'spm' in filename
        self._messages.append(Message(split_list[0].lower(), split_list[2].lower(), is_spam))
        file.close()

    def print_messages(self, with_body=True):
        for message in self._messages:
            print(message.msg_subject, message.msg_body if with_body else '')

    def message_bodies_as_string(self):
        return ''.join([message.msg_body for message in self._messages])

    def message_subject_as_string(self):
        return ''.join([message.msg_subject for message in self._messages])

    def get_messages(self):
        return self._messages
