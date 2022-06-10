import xml.etree.ElementTree as ET
import re
from abc import ABC, abstractmethod


class AbstractCreateNERData(ABC):
    """
    This abstract class defines an abstract method that includes
    the call to abstract operations to create objects reflecting the
    token-to-tag relationship for different challenges.
    """

    def __init__(self, file_list, annt_file_list=None):
        """ Annotation file list initialized to None to account
        for xml files that include both text and tags in one."""
        self.file_list = file_list
        self.annt_file_list = annt_file_list

    def create_ner_datasets(self) -> dict:
        """
        Skeleton of the steps from files to annotation dictionary.
        """
        data_dict = {}
        if self.annt_file_list is None:
            annt_list = self.file_list
        else:
            annt_list = self.annt_file_list
        for file, annt in zip(self.file_list, annt_list):
            # print(file)
            text = self.read_notes_file(file)
            text, event_labels = self.create_annt(annt, text)
            assert len(text) > 0 and len(event_labels) > 0, (
                "Something went wrong! Text and event labels could not be parsed")
            text, event_labels = self._merge_into_words(text, event_labels)
            for sentence, event_sentence in zip(text, event_labels):
                self._check_tags(event_sentence, sentence, file)
            data_dict[file.split('/')[-1]] = (text, event_labels)
        return data_dict

    @staticmethod
    def read_notes_file(file_name) -> list:
        """
        If not modified by subclasses it reads in a txt file
        with clinical notes and returns a word-level
        tokenization by sentence and the event labels initialized to 'O'.

        (Remark for n2c2 challenges: this method can be used as-is only for
        2009, 2010, 2011 challenge datasets. It needs to be modified for 2012
        and 2009 T2 challenges).

        :param file_name: str
        :return: list
        """
        with open(file=file_name, mode='r') as f:
            lines = f.readlines()
            text = [ll.strip('\n').split(' ') for ll in lines]
        return text

    def create_annt(self, file_name, text) -> (list, list):
        """
        Open annotation file if in txt format.

        :param file_name: str
        :param text: list
        :return: list, list
        """
        f = open(file=file_name, mode='r')
        text, event_labels = self.extract_tags(f.readlines(), text)
        f.close()
        return text, event_labels

    @abstractmethod
    def extract_tags(self, annotations, text) -> (list, list):
        pass

    @staticmethod
    def _replace_characters(obs_text, event_text) -> str:
        if '&apos;' in obs_text and '&apos;' not in event_text:
            event_text = event_text.replace("'", "&apos;")
        if '&quot;' in obs_text and '&quot;' not in event_text:
            event_text = event_text.replace('"', '&quot;')
        return event_text

    @staticmethod
    def _multiple_line_tag(event_labels,
                           text,
                           start_line,
                           end_line,
                           start_pos,
                           end_pos,
                           field_label) -> list:
        if end_line - start_line >= 1:
            for line in range(start_line, end_line + 1):
                t = text[line]
                if line == start_line:
                    if start_pos == len(t) - 1:
                        continue
                    else:
                        s = start_pos + 1
                else:
                    s = 0
                if line == end_line:
                    if end_pos == 0:
                        continue
                    else:
                        e = end_pos - 1
                else:
                    e = len(t)
                for i in range(s, e):
                    event_labels[line][i] = f"I-{':'.join(field_label)}"
        else:
            if end_pos - start_pos > 1:
                for i in range(start_pos + 1, end_pos):
                    event_labels[start_line][i] = f"I-{':'.join(field_label)}"
        return event_labels

    @staticmethod
    def _fix_extra_blanks(event_text, obs_text) -> (str, str):
        obs_text = list(obs_text)
        event_text = list(event_text)
        new_event_text = []
        for i in range(min(len(obs_text), len(event_text))):
            if event_text[i] == obs_text[i]:
                new_event_text.append(event_text[i])
            else:
                if event_text[i + 1] == obs_text[i]:
                    new_event_text.extend(event_text[i + 1:])
                    break
                else:
                    new_event_text = event_text
                    break
        obs_text = ''.join(obs_text)
        event_text = ''.join(new_event_text)
        return event_text, obs_text

    @staticmethod
    def _merge_into_words(text_by_char, all_labels_by_char) -> (list, list):
        assert len(text_by_char) == len(all_labels_by_char), "Incorrect # of sentences!"

        N = len(text_by_char)

        text_by_word, all_labels_by_word = [], []

        for sentence_num in range(N):
            sentence_by_char = text_by_char[sentence_num]
            labels_by_char = all_labels_by_char[sentence_num]

            assert len(sentence_by_char) == len(labels_by_char), "Incorrect # of chars in sentence!"
            S = len(sentence_by_char)

            if labels_by_char == (['O'] * len(sentence_by_char)):
                sentence_by_word = ''.join(sentence_by_char).split()
                labels_by_word = ['O'] * len(sentence_by_word)
            else:
                sentence_by_word, labels_by_word = [], []
                text_chunks, labels_chunks = [], []
                s = 0
                for i in range(S):
                    if i == S - 1:
                        text_chunks.append(sentence_by_char[s:])
                        labels_chunks.append(labels_by_char[s:])
                    elif labels_by_char[i] == 'O':
                        continue
                    else:
                        if i > 0 and labels_by_char[i - 1] == 'O':
                            text_chunks.append(sentence_by_char[s:i])
                            labels_chunks.append(labels_by_char[s:i])
                            s = i
                        if labels_by_char[i + 1] == 'O' or labels_by_char[i + 1][2:] != labels_by_char[i][2:]:
                            text_chunks.append(sentence_by_char[s:i + 1])
                            labels_chunks.append(labels_by_char[s:i + 1])
                            s = i + 1

                for text_chunk, labels_chunk in zip(text_chunks, labels_chunks):
                    assert len(text_chunk) == len(labels_chunk), "Bad Chunking (len)"
                    assert len(text_chunk) > 0, "Bad chunking (len 0)" + str(text_chunks) + str(labels_chunks)

                    labels_set = set(labels_chunk)
                    assert labels_set == {'O'} or (len(labels_set) <= 3 and 'O' not in labels_set), (
                        ("Bad chunking (contents) %s" % ', '.join(labels_set)) + str(text_chunks) + str(
                            labels_chunks),
                        f"Sentence: {sentence_by_word}",
                        f"Sentence labels: {labels_by_word}"
                    )

                    text_chunk_by_word = ''.join(text_chunk).split()
                    W = len(text_chunk_by_word)
                    if W == 0:
                        continue

                    if labels_chunk[0] == 'O':
                        labels_chunk_by_word = ['O'] * W
                    elif W == 1:
                        labels_chunk_by_word = [labels_chunk[0]]
                    elif W == 2:
                        labels_chunk_by_word = [labels_chunk[0], labels_chunk[-1]]
                    else:
                        labels_chunk_by_word = [
                                                   labels_chunk[0]
                                               ] + [labels_chunk[1]] * (W - 2) + [
                                                   labels_chunk[-1]
                                               ]

                    sentence_by_word.extend(text_chunk_by_word)
                    labels_by_word.extend(labels_chunk_by_word)

            assert len(sentence_by_word) == len(labels_by_word), "Incorrect # of words in sentence!"

            if len(sentence_by_word) == 0:
                continue

            text_by_word.append(sentence_by_word)
            all_labels_by_word.append(labels_by_word)
        return text_by_word, all_labels_by_word

    @staticmethod
    def _check_tags(event_sentence, sentence, file_name) -> None:
        i = 0
        while i < len(event_sentence):
            if re.match(r'B-', event_sentence[i]):
                start = i
                j = i + 1
                while j < len(event_sentence) and event_sentence[j] != 'O':
                    j += 1
                stop = j
                tag_start = event_sentence[i].split('-')[1]
                tag_set = set([event_sentence[idx].split('-')[1] for idx in range(start + 1, stop) if
                               event_sentence[idx].split('-')[1] != tag_start])
                if len(tag_set) > 1:
                    for t in tag_set:
                        assert f'B-{t}' in event_sentence[start:stop], (
                            f"File: {file_name} -- Check tag {event_sentence[start:stop]}/{sentence[start:stop]}")
                i = stop
            else:
                i += 1
        return


class CharacterBasedNERData(AbstractCreateNERData):
    """
    Subclass to process text and annotations at the character level.
    (Remark: for n2c2 challenge datasets this implements processing
    of 2012 and 2018 task).
    """

    def __init__(self, file_list, annt_file_list=None, xml_format=False):
        super(CharacterBasedNERData, self).__init__(file_list, annt_file_list)
        self.xml_format = xml_format

    def read_notes_file(self, file_name) -> list:
        # Read xml file
        with open(file_name, mode='r') as f:
            lines = f.readlines()
            text = []
            if self.xml_format:
                START_CDATA = "<TEXT><![CDATA["
                END_CDATA = "]]></TEXT>"
                in_text = False
                for i, ll in enumerate(lines):
                    if START_CDATA in ll:
                        text.append(list(ll[ll.find(START_CDATA) + len(START_CDATA):]))
                        in_text = True
                    elif END_CDATA in ll:
                        text.append(list(ll[:ll.find(END_CDATA)]))
                        break
                    elif in_text:
                        text.append(list(ll))
            else:
                text = [list(ll) for ll in lines]
        return text

    def create_annt(self, file_name, text) -> (list, list):
        if self.xml_format:
            try:
                xml_parsed = ET.parse(file_name)
            except:
                print("Cannot parse the document, returning empty lists.")
                return

            tag_containers = xml_parsed.findall('TAGS')
            assert len(tag_containers) == 1, "Found multiple tag sets!"
            tag_container = tag_containers[0]
            # Iterable object with
            event_tags = tag_container.findall('EVENT')
            text, event_labels = self.extract_tags(event_tags, text)
        else:
            text, event_labels = super().create_annt(file_name, text)
        return text, event_labels

    def extract_tags(self, event_tags, text) -> (list, list):
        pos_transformer = self._pos_transform(text)
        event_labels = [['O'] * len(sentence) for sentence in text]

        for event_tag in event_tags:
            if self.xml_format:
                base_label = event_tag.attrib["type"]
                polarity_label = event_tag.attrib["polarity"]
                modality_label = event_tag.attrib["modality"]
                field_label = [base_label, modality_label, polarity_label]
                start_pos, end_pos, event_text = event_tag.attrib['start'], event_tag.attrib['end'], event_tag.attrib[
                    'text']
                start_pos, end_pos = int(start_pos) + 1, int(end_pos)
                event_text = ' '.join(event_text.split())
            else:
                if re.match(r'^T[0-9]+', event_tag):
                    event_tag = event_tag.strip('\n')
                    event_tag = re.sub('\t', ' ', event_tag)
                    events = event_tag.split(' ')
                    try:
                        field_label, start_pos, end_pos, event_text = [events[1]], int(events[2]) + 1, int(
                            events[3]), ' '.join(events[4:]).strip()
                    except ValueError:
                        field_label, start_pos, end_pos, event_text = [events[1]], int(events[2]) + 1, int(
                            events[4]), ' '.join(events[5:]).strip()

            (start_line, start_char), (end_line, end_char) = pos_transformer[start_pos], pos_transformer[end_pos]
            # This part could be merged with _create_obs
            obs_text = []
            for line in range(start_line, end_line + 1):
                t = text[line]
                if line == start_line:
                    s = start_char
                else:
                    s = 0
                if line == end_line:
                    e = end_char
                else:
                    e = len(t)
                obs_text.append(''.join(t[s:e + 1]).strip())
            obs_text = ' '.join(obs_text)
            if self.xml_format:
                obs_text = ' '.join(obs_text.split())
            else:
                obs_text = ''.join(obs_text.strip())

            event_text = super()._replace_characters(obs_text, event_text)

            if not self.xml_format:
                event_text, obs_text = super()._fix_extra_blanks(event_text, obs_text)
            assert obs_text == event_text, (
                f"Texts don't match! ann::{event_text} vs note::{obs_text}",
                f"Start::{start_pos} -- End::{end_pos}; Line::{line}; Start_char::{s} -- End_char::{e}",
                f"Sentence::{t}")

            event_labels[end_line][end_char] = f"I-{':'.join(field_label)}"
            event_labels[start_line][start_char] = f"B-{':'.join(field_label)}"

            for line in range(start_line, end_line + 1):
                t = text[line]
                s = start_char + 1 if line == start_line else 0
                e = end_char - 1 if line == end_line else len(t) - 1
                for i in range(s, e + 1):
                    event_labels[line][i] = f"I-{':'.join(field_label)}"
        return text, event_labels

    @staticmethod
    def _pos_transform(chr_list) -> dict:
        pos_transformer = {}
        linear_pos = 1
        for line, sentence in enumerate(chr_list):
            for char_pos, char in enumerate(sentence):
                pos_transformer[linear_pos] = (line, char_pos)
                linear_pos += 1
        return pos_transformer


class TokenBasedNERData(AbstractCreateNERData):

    def __init__(self, file_list, annt_file_list):
        super(TokenBasedNERData, self).__init__(file_list, annt_file_list)

    def extract_tags(self, annotations, text) -> (list, list):
        event_labels = [['O'] * len(sentence) for sentence in text]
        for ll in annotations:
            ll = ll.split('||')
            # print(ll)
            field_and_event_text, start_pos, end_pos = [], -1, -1
            field_label = []
            i = 0
            if len(ll) > 3:
                while i < len(ll):
                    # print(i)
                    concept = ll[i]
                    # print(concept)
                    try:
                        field_and_event_text, start_pos, end_pos = concept.strip().split(' ')
                        # print(field_label, event_text)
                        # print(field_and_event_text, start_pos, end_pos)
                    except ValueError:
                        v = concept.strip().split(' ')
                        if len(v) < 4:
                            i += 1
                            continue
                        else:
                            field_and_event_text = ' '.join(v[0:len(v) - 2])
                            start_pos, end_pos = v[-2], v[-1]
                    field_label = [field_and_event_text.split('=', 1)[0].strip()]
                    event_labels = self.create_event_labels(field_and_event_text,
                                                            field_label,
                                                            start_pos,
                                                            end_pos,
                                                            text,
                                                            event_labels)
                    i += 1
            else:
                while i < len(ll):
                    concept = ll[i]
                    try:
                        field_and_event_text, start_pos, end_pos = concept.strip().split(' ')
                        # print(field_label, event_text)
                        # print(field_and_event_text, start_pos, end_pos)
                    except ValueError:
                        v = concept.strip().split(' ')
                        if len(v) < 4:
                            j = i
                            while j < len(ll):
                                tag, tag_text = ll[j].strip().split('=', 1)
                                tag_text = tag_text.strip('\n').strip('"')
                                # print(tag, tag_text)
                                # vector of concepts and assertions
                                field_label.append(tag_text)
                                j += 1
                            i = j
                        else:
                            field_and_event_text = ' '.join(v[0:len(v) - 2])
                            start_pos, end_pos = v[-2], v[-1]
                    i += 1
                event_labels = self.create_event_labels(field_and_event_text,
                                                        field_label,
                                                        start_pos,
                                                        end_pos,
                                                        text,
                                                        event_labels)
        return text, event_labels

    def create_event_labels(self, field_and_event_text, field_label,
                            start_pos, end_pos, text, event_labels):
        event_text = field_and_event_text.split('=', 1)[1].strip('"')
        event_text = event_text.strip('.').strip().strip(';').strip('.;').strip(':').lower()
        start_line, start_pos = start_pos.split(':')
        start_line, start_pos = int(start_line) - 1, int(start_pos)
        end_line, end_pos = end_pos.split(':')
        end_line, end_pos = int(end_line) - 1, int(end_pos)
        obs_text, line, t, s, e = self._create_obs(start_line, end_line, start_pos, end_pos, text)

        event_text = super()._replace_characters(obs_text, event_text)
        # print(field_label, event_text, obs_text, ll, t)
        if obs_text != event_text:
            obs_text, start_line, end_line = self._fix_unit_line_shift(start_line, end_line, start_pos, end_pos, text)
        assert obs_text == event_text, (
            f"Texts don't match! ann::{event_text} vs note::{obs_text}",
            f"Start::{start_pos} -- End::{end_pos}; Line::{line}; Start_char::{s} -- End_char::{e}",
            f"Sentence::{t}")
        event_labels[end_line][end_pos] = f"I-{':'.join(field_label)}"
        event_labels[start_line][start_pos] = f"B-{':'.join(field_label)}"
        event_labels = super()._multiple_line_tag(event_labels, text, start_line, end_line, start_pos,
                                                  end_pos, field_label)
        return event_labels

    def _fix_unit_line_shift(self, start_line, end_line, start_pos, end_pos, text):
        start_line = start_line - 1
        end_line = end_line - 1
        obs_text, line, t, s, e = self._create_obs(start_line, end_line, start_pos, end_pos, text)
        return obs_text, start_line, end_line

    @staticmethod
    def _create_obs(start_line, end_line, start_pos, end_pos, text):
        obs_text = []
        for line in range(start_line, end_line + 1):
            t = text[line]
            if line == start_line:
                s = start_pos
            else:
                s = 0
            if line == end_line:
                e = end_pos
            else:
                e = len(t)
            obs_text.append(' '.join(t[s:e + 1]).strip().lower())
        obs_text = ' '.join(obs_text).strip('.').strip(';').strip('.;').strip(':').strip()
        return obs_text, line, t, s, e

    @staticmethod
    def _merge_into_words(text_by_char, all_labels_by_char) -> (list, list):
        return text_by_char, all_labels_by_char


def ner_challenges(abstract_class=AbstractCreateNERData) -> dict:
    return abstract_class.create_ner_datasets()
