import xml.etree.ElementTree as ET
import re
from abc import ABC, abstractmethod


class AbstractCreateNER(ABC):
    """
    This abstract class defines an abstract method that includes
    the call to abstract operations to create objects reflecting the
    token-to-tag relationship for different challenges.
    """

    def template_method(self) -> None:
        """
        Skeleton of the steps from files to annotations.
        """
        self.read_notes_file()
        self.preprocessing()
        self.extract_tags()

    @staticmethod
    def read_notes_file(file_name) -> (list, list):
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
        event_labels = [['O'] * len(sentence) for sentence in text]
        return text, event_labels

    @abstractmethod
    def preprocessing(self) -> None:
        pass

    @abstractmethod
    def extract_tags(self) -> None:
        pass


START_CDATA = "<TEXT><![CDATA["
END_CDATA = "]]></TEXT>"


def read_xml_file(file_name, match_text=True):
    """
    Function that reads an xml file from the 2012 challenge
    and returns a list of lists of characters per sentence
    and a list of lists of character-level tags.

    :param file_name: path to the xml file
    :param match_text: whether to activate the match text check
    :return: list of lists of sentences by characters; list if lists of character-level tags
    """

    # Read xml file
    with open(file_name, mode='r') as f:
        lines = f.readlines()
        text, in_text = [], False
        for i, l in enumerate(lines):
            if START_CDATA in l:
                text.append(list(l[l.find(START_CDATA) + len(START_CDATA):]))
                in_text = True
            elif END_CDATA in l:
                text.append(list(l[:l.find(END_CDATA)]))
                break
            elif in_text:
                text.append(list(l))

    pos_transformer = _pos_transform(text)

    try:
        xml_parsed = ET.parse(file_name)
    except:
        print("Cannot parse the document, returning empty lists.")
        return [], []

    tag_containers = xml_parsed.findall('TAGS')
    assert len(tag_containers) == 1, "Found multiple tag sets!"
    tag_container = tag_containers[0]

    event_tags = tag_container.findall('EVENT')
    # One "O" per character (to merge later)
    event_labels = [['O'] * len(sentence) for sentence in text]

    for event_tag in event_tags:
        base_label = event_tag.attrib["type"]
        polarity_label = event_tag.attrib["polarity"]
        modality_label = event_tag.attrib["modality"]
        start_pos, end_pos, event_text = event_tag.attrib['start'], event_tag.attrib['end'], event_tag.attrib[
            'text']
        start_pos, end_pos = int(start_pos) + 1, int(end_pos)
        event_text = ' '.join(event_text.split())

        (start_line, start_char), (end_line, end_char) = pos_transformer[start_pos], pos_transformer[end_pos]

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
        obs_text = ' '.join(obs_text.split())

        if '&apos;' in obs_text and '&apos;' not in event_text:
            event_text = event_text.replace("'", "&apos;")
        if '&quot;' in obs_text and '&quot;' not in event_text:
            event_text = event_text.replace('"', '&quot;')

        if match_text:
            assert obs_text == event_text, (
                    ("Texts don't match! %s v %s" % (event_text, obs_text)) + '\n' + str((start_pos, end_pos,
                                                                                          line, s, e, t, file_name
                                                                                          )))

        event_labels[end_line][end_char] = f'I-{base_label}:{modality_label}:{polarity_label}'
        event_labels[start_line][start_char] = f'B-{base_label}:{modality_label}:{polarity_label}'

        for line in range(start_line, end_line + 1):
            t = text[line]
            s = start_char + 1 if line == start_line else 0
            e = end_char - 1 if line == end_line else len(t) - 1
            for i in range(s, e + 1):
                event_labels[line][i] = f'I-{base_label}:{modality_label}:{polarity_label}'
    return text, event_labels


def read_txt_file(txt_file_name, ann_file_name, match_text=True):
    """
    Function that reads both a txt file with raw text and an annotation file and returns the `text` and `event_labels`
    objects as the `read_xml_file` function.
    :param txt_file_name:
    :param ann_file_name:
    :param match_text:
    :return:
    """
    # Read txt file
    with open(txt_file_name, mode='r') as f:
        lines = f.readlines()
        # Tokenized sentences (word-level tokens)
        text = [ll.strip('\n').split(' ') for ll in lines]

    event_labels = [['O'] * len(sentence) for sentence in text]

    # Read annotation file (challenges 2009, 2010, 2011) -- line:token tags
    with open(ann_file_name, mode='r') as f:
        lines = f.readlines()
        for ll in lines:
            ll = ll.split('||')
            if len(ll) > 3:
                # 2009 challenge case
                for concept in ll:
                    try:
                        lab_and_text, start, stop = concept.split(' ')
                    except ValueError:
                        v = concept.split(' ')
                        if len(v) < 4:
                            continue
                        else:
                            lab_and_text = ' '.join(v[0:len(v) - 2])
                            start, stop = v[-2], v[-1]
                    field_label, field_text = lab_and_text.split('=', 1)[0].strip(), lab_and_text.split('=', 1)[
                        1].strip('"')
                    field_text = field_text.strip('.').strip().strip(';').strip('.;').strip(':').lower()
                    start_line, start_tkn = start.split(':')
                    start_line, start_tkn = int(start_line) - 1, int(start_tkn)
                    end_line, end_tkn = stop.split(':')
                    end_line, end_tkn = int(end_line) - 1, int(end_tkn)

                    obs_text = []
                    for line in range(start_line, end_line + 1):
                        t = text[line]
                        if line == start_line:
                            s = start_tkn
                        else:
                            s = 0
                        if line == end_line:
                            e = end_tkn
                        else:
                            e = len(t)
                        obs_text.append(' '.join(t[s:e + 1]).strip().lower())
                    obs_text = ' '.join(obs_text).strip('.').strip(';').strip('.;').strip(':').strip()

                    if '&apos;' in obs_text and '&apos;' not in field_text:
                        field_text = field_text.replace("'", "&apos;")
                    if '&quot;' in obs_text and '&quot;' not in field_text:
                        field_text = field_text.replace('"', '&quot;')

                    if match_text:
                        assert obs_text == field_text, (("Texts don't match! %s v %s" % (field_text, obs_text)) + '\n' +
                                                        str((start_tkn, end_tkn,
                                                             line, s, e, t, ann_file_name
                                                             )))
                    event_labels[end_line][end_tkn] = f'I-{field_label}'
                    event_labels[start_line][start_tkn] = f'B-{field_label}'
                    if end_line - start_line >= 1:
                        for line in range(start_line, end_line + 1):
                            t = text[line]
                            if line == start_line:
                                if start_tkn == len(t) - 1:
                                    continue
                                else:
                                    s = start_tkn + 1
                            else:
                                s = 0
                            if line == end_line:
                                if end_tkn == 0:
                                    continue
                                else:
                                    e = end_tkn - 1
                            else:
                                e = len(t)
                            for i in range(s, e):
                                event_labels[line][i] = f'I-{field_label}'
                    else:
                        if end_tkn - start_tkn > 1:
                            for i in range(start_tkn + 1, end_tkn):
                                event_labels[start_line][i] = f'I-{field_label}'
            else:
                field_label = []
                for concept in ll:
                    try:
                        # Concept
                        lab_and_text, start, stop = concept.split(' ')
                    except ValueError:
                        v = concept.split(' ')
                        # Concept tags
                        if len(v) < 4:
                            tag, tag_text = concept.split('=', 1)
                            tag_text = tag_text.strip('\n').strip('"')
                            if tag_text != "nm":
                                # vector of concepts and assertions
                                field_label.append(tag_text)
                        else:
                            # Multi-word concept
                            lab_and_text = ' '.join(v[0:len(v) - 2])
                            start, stop = v[-2], v[-1]

                    # 2009 challenge case

                    field_text = lab_and_text.split('=', 1)[1].strip('"')
                    field_text = field_text.strip('.').strip().strip(';').strip('.;').strip(':').lower()
                    start_line, start_tkn = start.split(':')
                    start_line, start_tkn = int(start_line) - 1, int(start_tkn)
                    end_line, end_tkn = stop.split(':')
                    end_line, end_tkn = int(end_line) - 1, int(end_tkn)

                    obs_text = []
                    for line in range(start_line, end_line + 1):
                        t = text[line]
                        if line == start_line:
                            s = start_tkn
                        else:
                            s = 0
                        if line == end_line:
                            e = end_tkn
                        else:
                            e = len(t)
                        obs_text.append(' '.join(t[s:e + 1]).strip().lower())
                    obs_text = ' '.join(obs_text).strip('.').strip(';').strip('.;').strip(':').strip()

                    if '&apos;' in obs_text and '&apos;' not in field_text:
                        field_text = field_text.replace("'", "&apos;")
                    if '&quot;' in obs_text and '&quot;' not in field_text:
                        field_text = field_text.replace('"', '&quot;')

                    if match_text:
                        # Case in which row is shifted by one
                        if obs_text != field_text:
                            obs_text = []
                            start_line = start_line - 1
                            end_line = end_line - 1
                            for line in range(start_line, end_line + 1):
                                t = text[line]
                                if line == start_line:
                                    s = start_tkn
                                else:
                                    s = 0
                                if line == end_line:
                                    e = end_tkn
                                else:
                                    e = len(t)
                                obs_text.append(' '.join(t[s:e + 1]).strip().lower())
                            obs_text = ' '.join(obs_text).strip('.').strip(';').strip('.;').strip(':').strip()
                    assert obs_text == field_text, (("Texts don't match! %s v %s" % (field_text, obs_text)) + '\n' +
                                                    str((start_tkn, end_tkn,
                                                         line + 1, s, e, t, ann_file_name
                                                         )))

                field_label = ':'.join(field_label).strip('\n')
                event_labels[end_line][end_tkn] = f'I-{field_label}'
                event_labels[start_line][start_tkn] = f'B-{field_label}'
                if end_line - start_line >= 1:
                    for line in range(start_line, end_line + 1):
                        t = text[line]
                        if line == start_line:
                            if start_tkn == len(t) - 1:
                                continue
                            else:
                                s = start_tkn + 1
                        else:
                            s = 0
                        if line == end_line:
                            if end_tkn == 0:
                                continue
                            else:
                                e = end_tkn - 1
                        else:
                            e = len(t)
                        for i in range(s, e):
                            event_labels[line][i] = f'I-{field_label}'
                else:
                    if end_tkn - start_tkn > 1:
                        for i in range(start_tkn + 1, end_tkn):
                            event_labels[start_line][i] = f'I-{field_label}'

    return text, event_labels


# 2018 T2
def read_txt_file2(file_name, ann_file_name, match_text=True):
    """
    Function that reads a txt file from the 2018 T2 challenge
    and returns a list of lists of characters per sentence
    and a list of lists of character-level tags.

    :param file_name: path to the note file
    :param ann_file_name: path to the annotation file
    :param match_text: whether to activate the match text check
    :return: list of lists of sentences by characters; list if lists of character-level tags
    """

    # Read txt file
    with open(file_name, mode='r') as f:
        lines = f.readlines()
        text = []
        for i, l in enumerate(lines):
            text.append(list(l))

    pos_transformer = _pos_transform(text)
    event_labels = [['O'] * len(sentence) for sentence in text]

    with open(ann_file_name, mode='r') as f:
        lines = f.readlines()
        for line in lines:
            if re.match(r'^T[0-9]+', line):
                line = re.sub('\t', ' ', line.strip('\n'))
                fields = line.split(' ')
                try:
                    field_label, start_pos, end_pos, field_text = fields[1], int(fields[2]) + 1, int(
                        fields[3]), ' '.join(fields[4:]).strip()
                except ValueError:
                    field_label, start_pos, end_pos, field_text = fields[1], int(fields[2]) + 1, int(
                        fields[4]), ' '.join(fields[5:]).strip()

                (start_line, start_char), (end_line, end_char) = pos_transformer[start_pos], pos_transformer[end_pos]
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

                if '&apos;' in obs_text and '&apos;' not in field_text:
                    field_text = field_text.replace("'", "&apos;")
                if '&quot;' in obs_text and '&quot;' not in field_text:
                    field_text = field_text.replace('"', '&quot;')

                if match_text:
                    assert obs_text == field_text, (
                            ("Texts don't match! %s v %s" % (field_text, obs_text)) + '\n' + str((start_pos, end_pos,
                                                                                                  line, s, e, t,
                                                                                                  file_name
                                                                                                  )))
                event_labels[end_line][end_char] = f'I-{field_label}'
                event_labels[start_line][start_char] = f'B-{field_label}'

                for line in range(start_line, end_line + 1):
                    t = text[line]
                    s = start_char + 1 if line == start_line else 0
                    e = end_char - 1 if line == end_line else len(t) - 1
                    for i in range(s, e + 1):
                        event_labels[line][i] = f'I-{field_label}'
    return text, event_labels


"""
Private functions
"""


# def _read_file():

def _pos_transform(chr_list):
    pos_transformer = {}
    linear_pos = 1
    for line, sentence in enumerate(chr_list):
        for char_pos, char in enumerate(sentence):
            pos_transformer[linear_pos] = (line, char_pos)
            linear_pos += 1
    return pos_transformer
