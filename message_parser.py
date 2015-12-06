import os


class Message:
    def __init__(self, subject, msg):
        self.msg_subject = subject
        self.msg_body = msg


class MessageParser:
    def __init__(self):
        self.path = '/home/leopold/Documents/spam_examples/lingspam_public/bare'
        self._messages = []
        # bare , lemm_stop, lemm, stop

    def init_sub_files(self):
        for dir_path, dir_names, file_names, in os.walk(self.path):
            for filename in file_names:
                try:
                    self.parse_file(dir_path, filename)
                except IndexError as e:
                    print('Index except: ', e.args)
        print('Proceed %d messages.' % len(self._messages))

    def parse_file(self, dir_path, filename):
        file = open(dir_path + '/' + filename, 'r')
        text = file.read()
        split_list = text.split(sep='\n')
        if len(split_list) != 4:
            file.close()
            raise IndexError('split list wrong size:', len(split_list), ' !')
        self._messages.append(Message(split_list[0], split_list[2]))
        file.close()

    def get_messages(self):
        return self._messages


if __name__ == '__main__':
    message_parser = MessageParser()
    message_parser.init_sub_files()
