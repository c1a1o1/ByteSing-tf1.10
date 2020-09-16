'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
'''
from . import cmudict

_pad        = '_'
_eos        = '~'
_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'\"(),-.:;? %/'
_digits     = '0123456789'
_all_phonemes = ['b', 'p', 'f', 'm', \
			'd', 't', 'n', 'l', \
            'g', 'k', 'h', \
            'j', 'q', 'x', \
            'zh', 'ch', 'sh', 'r', \
            'z', 'c', 's',\
            'a',  'ai', 'ao',  'an',  'ang', \
            'o',  'ou', 'ong', \
            'e',  'ei', 'en',  'eng', 'er', 'ev', \
            'i',  'ix', 'iii', \
            'ia', 'iao','ian', 'iang','ie', \
            'in', 'ing','io',  'iou', 'iong', \
            'u',  'ua', 'uo',  'uai', 'uei', \
            'uan','uen','uang','ueng', \
            'v',  've', 'van', 'vn', \
            'ng', 'mm', 'nn',\
            'rr', 'sp']
_all_pitch = ['0','50','51','52','53','54','55','56','57','58','59',\
			'60','61','62','63','64','65','66','67','68','69',\
			'70','71','72','73','74','75','76','77','78','79','80']
_phoneme_type = ['0', '1', '2']

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
#_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = [_pad, _eos] + list(_characters) + list(_digits) #+ _arpabet
duration_symbols = [_all_phonemes, _all_pitch, _phoneme_type]

