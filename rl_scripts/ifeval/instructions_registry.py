# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Registry of all instructions."""

from . import instructions

_PARAGRAPH = "paragraphs:"

_KEYWORD = "keywords:"

_LETTER = "letters:"

_LANGUAGE = "language:"

_LENGTH = "length_constraints:"

_CONTENT = "detectable_content:"

_FORMAT = "detectable_format:"

_MULTITURN = "multi-turn:"

_COMBINATION = "combination:"

_STARTEND = "startend:"

_CHANGE_CASES = "change_case:"

_PUNCTUATION = "punctuation:"

_NEW = "new:"

_COPY = "copy:"

_BASIC = "basic:"

_FIRSTWORD = "first_word:"

_LASTWORD = "last_word:"

_COUNT = "count:"


FUNCTION_DICT = {
    # IFEval Constraints
    _KEYWORD + "existence": instructions.KeywordChecker,
    _KEYWORD + "frequency": instructions.KeywordFrequencyChecker,
    _KEYWORD + "forbidden_words": instructions.ForbiddenWords,
    _KEYWORD + "letter_frequency": instructions.LetterFrequencyChecker,
    _LANGUAGE + "response_language": instructions.ResponseLanguageChecker,
    _LENGTH + "number_sentences": instructions.NumberOfSentences,
    _LENGTH + "number_paragraphs": instructions.ParagraphChecker,
    _LENGTH + "number_words": instructions.NumberOfWords,
    _LENGTH + "nth_paragraph_first_word": instructions.ParagraphFirstWordCheck,
    _CONTENT + "number_placeholders": instructions.PlaceholderChecker,
    _CONTENT + "postscript": instructions.PostscriptChecker,
    _FORMAT + "number_bullet_lists": instructions.BulletListChecker,
    _FORMAT + "constrained_response": instructions.ConstrainedResponseChecker,
    _FORMAT + "number_highlighted_sections": instructions.HighlightSectionChecker,
    _FORMAT + "multiple_sections": instructions.SectionChecker,
    _FORMAT + "json_format": instructions.JsonFormat,
    _FORMAT + "title": instructions.TitleChecker,
    _COMBINATION + "two_responses": instructions.TwoResponsesChecker,
    _COMBINATION + "repeat_prompt": instructions.RepeatPromptThenAnswer,
    _STARTEND + "end_checker": instructions.EndChecker,
    _CHANGE_CASES + "capital_word_frequency": instructions.CapitalWordFrequencyChecker,
    _CHANGE_CASES + "english_capital": instructions.CapitalLettersEnglishChecker,
    _CHANGE_CASES + "english_lowercase": instructions.LowercaseLettersEnglishChecker,
    _PUNCTUATION + "no_comma": instructions.CommaChecker,
    _STARTEND + "quotation": instructions.QuotationChecker,
    # New Constraints
    _COPY + "repeat_phrase": instructions.RepeatPhraseChecker,
    _COPY + "copy": instructions.CopyChecker,
    _NEW + "copy_span_idx": instructions.CopySpanIdxChecker,
    _FORMAT + "sentence_hyphens": instructions.SentenceHyphenChecker,
    _KEYWORD + "no_adjacent_consecutive": instructions.AdjacentLetterChecker,
    _FORMAT + "square_brackets": instructions.SquareBracketChecker,
    _KEYWORD + "word_once": instructions.KeywordFrequencyOnceChecker,
    _KEYWORD + "word_count_different_numbers": instructions.KeywordFrequencyCheckerDifferent,
    _KEYWORD + "exclude_word_harder": instructions.ExcludeWordHarderChecker,
    _PARAGRAPH + "paragraphs": instructions.ParagraphBasicChecker,
    _PARAGRAPH + "paragraphs2": instructions.ParagraphBasicChecker2,
    _FIRSTWORD + "first_word_sent": instructions.FirstWordSentChecker,
    _FIRSTWORD + "first_word_answer": instructions.FirstWordAnswerChecker,
    _LASTWORD + "last_word_sent": instructions.LastWordSentChecker,
    _LASTWORD + "last_word_answer": instructions.LastWordAnswerChecker,
    _FORMAT + "bigram_wrapping": instructions.BiGramWrappingChecker,
    _COPY + "copying_simple": instructions.CopyingSimpleChecker,
    _COPY + "copying_multiple": instructions.CopyingMultipleChecker,
    _PUNCTUATION + "punctuation_dot": instructions.PunctuationDotChecker,
    _PUNCTUATION + "punctuation_exclamation": instructions.PunctuationExclamationChecker,
    _COUNT + "lowercase_counting": instructions.LowercaseCountingChecker,
    _LETTER + "letter_counting": instructions.LetterCountingChecker,
    _LETTER + "letter_counting2": instructions.LetterFrequencyChecker,
    _COUNT + "counting_composition": instructions.CountingCompositionChecker,
    _COUNT + "count_unique": instructions.CountUniqueChecker,
    _COUNT + "count_increment_word": instructions.CountIncrementWordChecker,
    _KEYWORD + "palindrome": instructions.PalindromeBasicChecker,
    _KEYWORD + "keyword_specific_position": instructions.KeywordSpecificPositionChecker,
    _KEYWORD + "start_end": instructions.StartEndChecker,
}

INSTRUCTION_DICT = FUNCTION_DICT
