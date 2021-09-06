'use strict';

// Create a story div and highlight the words
function HighlightStoryView(_story, _matchedWords, _mode) {
    const story = _story;
    const matchedWords = _matchedWords;
    const mode = _mode;
    // The container
    let container = $('<div></div>');
    this.getContainer = function () {
        return container;
    };

    // Get the pattern
    function getWordPatternV2(targetWord, taggedNLPWords) {
        let pattern = '';
        // Get the previous word
        let previousWord = taggedNLPWords.find(taggedWord =>
            taggedWord['sentId'] === targetWord['sentId'] && taggedWord['id'] === targetWord['id'] - 1
        );

        if (previousWord) {
            if (previousWord['ud'] === 'punct') {
                pattern = '(?<=\\' + previousWord['word'] + '\\s*)';
            } else {
                pattern = '(?<=' + previousWord['word'] + '\\s*)';
            }
        }
        pattern += targetWord['word']

        // Get the next word and use 'Lookbehind assertion' regular expression
        // x(?=y) Matches "x" only if "x" is preceded by "y".
        // Ref: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Regular_Expressions/Assertions
        let nextWord = taggedNLPWords.find(taggedWord =>
            taggedWord['sentId'] === targetWord['sentId'] && taggedWord['id'] === targetWord['id'] + 1
        );


        if (nextWord) {// Match the word using lookahead assertion
            if (nextWord['ud'] === 'punct') {
                // Escape char before the punctuation is needed
                pattern += '\\s*\\' + nextWord['word'];
            } else {
                // Check if the next word is like an abbreviation such as 'd 've
                if (nextWord['word'].includes("'")) {
                    // Match the word (without white space before the next word) using lookahead assertion
                    pattern += '(?=' + nextWord['word'] + ')';// Space before the next word is needed
                } else {
                    // Match the word using lookahead assertion
                    pattern += '(?=\\s*' + nextWord['word'] + ')';// Space before the next word is needed
                }
            }
        }
        return pattern;
    }

    // Highlight a word with NRC color palette or subject
    function highlightWord(targetWord, className, storyDiv, taggedNLPWords) {
        let pattern = getWordPatternV2(targetWord, taggedNLPWords);
        // console.log(pattern);
        try {
            let regex = new RegExp(pattern, 'i');
            // Add Bootstrap tooltip
            storyDiv.markRegExp(regex, {
                "className": className,
                "ignoreGroups": 0,
                "acrossElements": true, // Search through the element so that "supermarket owner" can be highlight
                "noMatch": function (term) {
                    console.error(term + " has no match.");
                    // term is the not found term
                },
                "filter": function (textNode, foundTerm, totalCounter) {
                    // textNode is the text node which contains the found term
                    // foundTerm is the found search term
                    // totalCounter is a counter indicating the total number of all marks
                    //              at the time of the function call
                    // console.log("foundTerm = ", foundTerm, " totalCounter = ", totalCounter);
                    return true; // must return either true or false
                },
                "each": function (node) {
                    // let title = taggedNLPWord['word'] + '_' + taggedNLPWord['pos'] + ': ' + taggedNLPWord['ud']
                    //     + '(' + taggedNLPWord['head'] + ', ' + taggedNLPWord['word'] + '-' + taggedNLPWord['id'] + ')';
                    // // Add tooltip to each element, e.g. data-toggle="tooltip" data-placement="top" title="Tooltip on top"
                    // node.setAttribute("data-toggle", "tooltip");
                    // node.setAttribute("data-html", "true");
                    // node.setAttribute("title", title);
                },
                done: function () {

                }
            });
        } catch (e) {
            console.error("error:" + pattern);
        }

    }


    // Highlight a word
    function highlightWordOnly(word, className, storyDiv) {
        storyDiv.mark(word, {
            "className": className,
            "accuracy": {
                "value": "exactly",
                "limiters": [",", "."]
            },
            "noMatch": function (term) {
                console.error("Can not highlight " + term);
            },
        });
    }


    // Highlight the dependent/head word goes through each nlp_tagged result of the story
    // The words that has dependencies to the subject: 1) the word is the subject 2) the word whose head word is the subject
    // The noun phrases are also highlight.
    function _createUI() {

        // Story div displays the texts and highlights the key words
        let storyDiv = new TextView(story);// Tag the story text with entities
        let storyContainer = storyDiv.getContainer();
        // let taggedNLPWords = story['taggedWords'];

        // // Highlight the dependent words w.r.t. color pallet suggested by NRC
        // for (let word of matchedWords) {
        //     highlightWord(word, "keyword", storyContainer, taggedNLPWords);
        // }
        // if (mode === 'noun') {
        //     // Highlight the noun phrases
        //     let nouns = matchedWords.filter(w => w['pennPOS'].startsWith('nn'));
        //     for (let noun of nouns) {
        //         let nounPhrase = Utility.getLongestNounPhrase(noun, story);
        //         if (nounPhrase) {
        //             for (let word of nounPhrase) {
        //                 if (!matchedWords.some(w => w['sentId'] === word['sentId'] && w['id'] === word['id'])) {
        //                     highlightWord(word, "phrase", storyContainer, taggedNLPWords);
        //                 }
        //             }
        //         }
        //         // console.log(nounPhrase);
        //     }
        // } else if (mode === 'phrase') {
        //     // Highlight the noun phrases
        //     let verbs = matchedWords.filter(w => w['pennPOS'].startsWith('vb'));
        //     let subVerb = verbs[verbs.length - 1];
        //     let verbPhrase = Utility.getShortestVerbPhrase(subVerb, story);
        //     if (verbPhrase) {
        //         for (let word of verbPhrase) {
        //             if (!matchedWords.some(w => w['sentId'] === word['sentId'] && w['id'] === word['id'])) {
        //                 highlightWord(word, "phrase", storyContainer, taggedNLPWords);
        //             }
        //         }
        //     }
        // } else if (mode === 'clause') {
        //     // keywords and triple words
        //     let {keyword, tripleWords} = Utility.getMatchedTriple(story);
        //     if (tripleWords) {
        //         for (let word of tripleWords) {
        //             if (!matchedWords.some(w => w['sentId'] === word['sentId'] && w['id'] === word['id'])) {
        //                 highlightWord(word, "phrase", storyContainer, taggedNLPWords);
        //             }
        //         }
        //     }
        // }
        //
        // // Highlight entities
        // storyDiv.highlightEntities();

        container.append(storyContainer);
    }

    _createUI();
}
