import music21 as m21
import pinyin as py
data = m21.converter.parse('./myData/001.musicxml')


lines = []
part = data.parts.flat
for i in range(len(part.notesAndRests)):
    event = part.notesAndRests[i]
    if isinstance(event, m21.note.Note):
        duration = event.seconds
        # nameWithOctave = event.pitch.nameWithOctave
        # frequency = event.pitch.frequency
        midi = event.pitch.midi
        # diatonicNoteNum = event.pitch.diatonicNoteNum
        if len(event.lyrics) > 0:
            token = event.lyrics[1].text+'3'
            token = py.split_pinyin(token)
            if token[0] != '':
                # lines.append(str(duration) + '|' + str(midi) + '|' + '0' + '|' + token[0])
                # lines.append(str(duration) + '|' + str(midi) + '|' + '1' + '|' + token[1])
                lines.append([duration, midi, 0, token[0]])
                lines.append([duration, midi, 1, token[1]])
            elif token[1] != '':
                lines.append([duration, midi, 2, token[1]])
        else:
            temp = lines[-1]
            lines[-1][0] = lines[-1][0] + duration         
        # token = event.lyrics[1].text+'3' if len(event.lyrics) > 0 else "<PAD>"
        # print(duration, nameWithOctave, midi, diatonicNoteNum, token)
    elif isinstance(event, m21.note.Rest):
        duration = event.seconds
        midi = 0
        token = "sp"
        if lines[-1][-1] != 'sp':
            lines.append([duration, midi, 2, token])
        else:
            lines[-1][0] = lines[-1][0] + duration
        # print(duration, frequency, token)

with open('./myData/train.txt', 'w') as f:
    for line in lines:
        f.writelines(str(line[0]) + '|' + str(line[1]) + '|' + str(line[2]) + '|' + line[3] + '\n')

