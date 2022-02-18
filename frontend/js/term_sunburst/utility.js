// Utility functions
class Utility{

    // Collect top 5 frequent from key phrases
    static collect_title_words(key_phrases, docs){
        // Create word frequency list
        const create_word_freq_list = function(key_phrases){
            // Create bi-grams
            const create_bi_grams = function(words){
                const bi_grams = [];
                if(words.length === 2){
                    bi_grams.push(words[0] + " " + words[1]);
                }else if(words.length === 3) {
                    bi_grams.push(words[1] + " " + words[2]);
                }
                return bi_grams;
            };
            // Get the docs containing the word
            const get_word_docs = function(word){
                let doc_ids = [];
                for(const doc of docs){
                    const key_phrases = doc['KeyPhrases'];
                    const found = key_phrases.find(p => p.toLowerCase().includes(word))
                    if(found){
                        doc_ids.push(doc['DocId']);
                    }
                }
                return doc_ids;
            }

            let word_freq_list = [];
            // Collect word frequencies from the list of key phrases.
            for(const key_phrase of key_phrases){
                const words = key_phrase.split(" ");
                const bi_grams = create_bi_grams(words);
                const n_grams = words.concat(bi_grams);
                // collect uni_gram
                for (const n_gram of n_grams){
                    const range = n_gram.split(" ").length;
                    const found_word = word_freq_list.find(w => w['word'].toLowerCase() === n_gram.toLowerCase())
                    if(found_word){
                        found_word['freq'] += 1;
                    }else{
                        const doc_ids = get_word_docs(n_gram.toLowerCase());
                        if(n_gram === n_gram.toUpperCase()){
                            word_freq_list.push({'word': n_gram, 'freq':1, 'range': range, 'doc_ids':doc_ids});
                        }else{
                            word_freq_list.push({'word': n_gram.toLowerCase(), 'freq':1, 'range': range, 'doc_ids': doc_ids});
                        }
                    }
                }
            }

            return word_freq_list;
            // console.log(word_freq_list);
        }

        const word_freq_list = create_word_freq_list(key_phrases);

        // Update top word frequencies and pick up top words that increase the maximal coverage
        const pick_top_words= function(top_words, candidate_words, top_n){
            // Get all the words
            let all_words = top_words.concat(candidate_words);
            // Go through top_words and check if any top word has two words (such as 'twitter data'
            // If so, reduce the frequency of its single word (freq('twitter') -1 and freq('data') -1)
            for(const top_word of top_words){
                if(top_word['range'] === 2){
                    // Get the single word from a bi-gram
                    const top_word_list = top_word['word'].split(" ");
                    // Go through each individual word of top_word
                    for(const individual_word of top_word_list){
                        // Update the frequencies of the single word
                        for(const word of all_words){
                            if(word['word'].toLowerCase() === individual_word.toLowerCase()){
                                // Update the freq
                                word['freq'] = word['freq'] - top_word['freq'];
                            }
                        }
                    }
                }
            }
            // Remove the words of no frequencies
            all_words = all_words.filter(w => w['freq'] > 0);
            // Sort all the words by frequencies
            all_words.sort((a, b) => {
                if(b['freq'] === a['freq']){
                    return b['range'] - a['range'];     // Prefer longer phrases
                }
                return b['freq'] - a['freq'];
            });
            // console.log("All words", all_words.map(w => w['word'] + '(' + w['freq'] + ')'));
            // Pick up top frequent word from all_words
            const new_top_words = all_words.slice(0, top_n);
            const new_candidate_words = all_words.slice(top_n);
            return [new_top_words, new_candidate_words];
        }

        // Pick up top 5 frequent words
        const top_n = 6;
        // Sort by freq
        word_freq_list.sort((a, b) => {
            if(b['freq'] === a['freq']){
                return b['range'] - a['range'];     // Prefer longer phrases
            }
            return b['freq'] - a['freq'];
        });
        console.log("word_freq_list", word_freq_list);
        // Select top 5 bi_grams as default top_words
        let top_words = word_freq_list.slice(0, top_n);
        let candidate_words = word_freq_list.slice(top_n);
        let is_same = false;
        for(let iteration=0; !is_same && iteration<10; iteration++){
            // console.log("Group id: ", group_id);
            console.log("Iteration: ", iteration);
            console.log("top_words:", top_words.map(w => w['word'] + '(' + w['freq'] + ')'));
            const [new_top_words, new_candidate_words] = pick_top_words(top_words, candidate_words, top_n);
            console.log("new top words: ", new_top_words.map(w => w['word'] + '(' + w['freq'] + ')'));
            // Check if new and old top words are the same
            is_same = true;
            for(const new_word of new_top_words){
                const found = top_words.find(w => w['word'] === new_word['word']);
                if(!found){
                    is_same = is_same && false;
                }
            }
            top_words = new_top_words;
            candidate_words = new_candidate_words;
        }

        // Return the top 3
        return top_words.slice(0, 5).map(w => w['word']);
    }

    // Create a dict to store the relation between word and doc
    static create_word_doc_dict(title_words, docs, word_key_phrase_dict){
        let dict = {};
        for(const [title_word, key_phrases] of Object.entries(word_key_phrase_dict)){
            let word_doc_ids = [];
            // Match the key phrases with title words and assign key phrase
            for(const key_phrase of key_phrases) {
                // // Collect the doc containing the key phrase
                const relevant_docs = docs.filter(d => {
                    const found = d['KeyPhrases'].find(kp => kp.toLowerCase() === key_phrase.toLowerCase());
                    if(found){
                        return true;
                    }
                    return false;
                });
                for (const doc of relevant_docs){
                    if(!word_doc_ids.includes(doc['DocId'])){
                        word_doc_ids.push(doc['DocId']);
                    }
                }
            }
            dict[title_word] = word_doc_ids;
        }
        return dict;
    }

    // Create a dict of word and key phrases
    // Allocate the key phrases based on the words
    static create_word_key_phrases_dict(title_words, key_phrases){
        let dict = {};
        // Initialise dict with an array
        for(const title_word of title_words){
            dict[title_word] = [];
        }
        // Add misc
        dict['others'] = [];
        // Match the key phrases with title words and assign key phrase
        for(const key_phrase of key_phrases) {
            let is_found = false;
            for(const title_word of title_words){
                if(key_phrase.toLowerCase().includes(title_word.toLowerCase())){
                    dict[title_word].push(key_phrase);
                    is_found = true;
                }
            }
            // If not found, assign to 'misc'
            if(!is_found){
                dict['others'].push(key_phrase);
            }
        }
        return dict;
    }

    // Collect the phrase to doc relation
    static collect_phrase_docs(group_docs) {
        let dict = {};
        for (const doc of group_docs) {
            const doc_id = doc['DocId'];
            const key_phrases = doc['KeyPhrases'];
            for (const key_phrase of key_phrases) {
                if (key_phrase.toLowerCase() in dict) {
                    dict[key_phrase.toLowerCase()].push(doc_id);
                } else {
                    dict[key_phrase.toLowerCase()] = [doc_id];
                }
            }
        }
        return dict;
    }



}



// const pick_top_words= function(top_words, candidate_words, top_n){
//     // Get all the words
//     let all_words = top_words.concat(candidate_words);
//     // Go through top_words and check if any top word has two words (such as 'twitter data'
//     // If so, reduce the frequency of its single word (freq('twitter') -1 and freq('data') -1)
//     for(const top_word of top_words){
//         if(top_word['range'] === 2){
//             // Get the single word from a bi-gram
//             const top_word_list = top_word['word'].split(" ");
//             // Go through each individual word of top_word
//             for(const individual_word of top_word_list){
//                 // Update the frequencies of the single word
//                 for(const word of all_words){
//                     if(word['word'].toLowerCase() === individual_word.toLowerCase()){
//                         // Update the freq
//                         word['freq'] = word['freq'] - top_word['freq'];
//                     }
//                 }
//             }
//         }
//     }
//     // Remove the words of no frequencies
//     all_words = all_words.filter(w => w['freq'] > 0);
//     // Sort all the words by frequencies
//     all_words.sort((a, b) => {
//         if(b['freq'] === a['freq']){
//             return b['range'] - a['range'];     // Prefer longer phrases
//         }
//         return b['freq'] - a['freq'];
//     });
//     // console.log("All words", all_words.map(w => w['word'] + '(' + w['freq'] + ')'));
//     // Pick up top frequent word from all_words
//     const new_top_words = all_words.slice(0, top_n);
//     const new_candidate_words = all_words.slice(top_n);
//     return [new_top_words, new_candidate_words];
// }

// // Collect all key phrases co-occurring with title words
// function collect_word_neighbour_phrases() {
//     // Collect the relation between key phrases and its neighbour key phrases
//     const collect_other_key_phrases_by_phrase = function () {
//         let dict = {};
//         for (const key_phrase of group_key_phrases) {
//             // Get the docs containing key phrase
//             // Get the neighbour phrases
//             let neighbour_phrases = [];
//             for (const doc of group_docs) {
//                 const doc_key_phrases = doc['KeyPhrases'];
//                 const found = doc_key_phrases.find(kp => kp.toLowerCase() === key_phrase.toLowerCase());
//                 if (found) {
//                     const other_phrases = doc_key_phrases.filter(kp => kp.toLowerCase() !== key_phrase.toLowerCase());
//                     neighbour_phrases = neighbour_phrases.concat(other_phrases);
//                 }
//             }
//             dict[key_phrase.toLowerCase()] = neighbour_phrases;
//         }
//         return dict;
//     };
//
//     // Extract the common words from a list of key phrases
//     const extract_words_from_phrases = function (phrases) {
//         let neighboring_words = [];
//         for (const phrase of phrases) {
//             const words = phrase.toLowerCase().split(" ");
//             const last_word = words[words.length - 1];
//             const found = neighboring_words.find(w => w['word'] === last_word);
//             if (!found) {
//                 neighboring_words.push({'word': last_word, 'phrases': [phrase]})
//             } else {
//                 found['phrases'].push(phrase);
//             }
//         }
//         // Sort by the number of phrases
//         neighboring_words.sort((a, b) => b['phrases'].length - a['phrases'].length);
//         return neighboring_words;
//     };
//
//     // A dict stores the relation between a key phrases and its neighbouring phrases
//     const other_phrases_dict = collect_other_key_phrases_by_phrase();
//     let dict = {};
//     for (const title_word of title_words) {
//         // Keep top 10 neighbouring words
//
//         // Get the key phrases about title word
//         const key_phrases = word_key_phrase_dict[title_word];
//         let neighboring_phrases = [];
//         // Aggregate all neighbouring key phrases of key phrases
//         for (const key_phrase of key_phrases) {
//             // Get the neighbouring key phrases
//             const other_phrases = other_phrases_dict[key_phrase.toLowerCase()].map(kp => kp.toLowerCase());
//             neighboring_phrases = neighboring_phrases.concat(other_phrases.map(p => p.toLowerCase()));
//
//             // dict[title_word].push({'phrase': key_phrase, 'neighboring_words': neighboring_words});
//         }
//         // Remove duplicated
//         neighboring_phrases = [...new Set(neighboring_phrases)];
//         // Obtain the common words from a list of neighbouring phrases
//         let neighbouring_words = extract_words_from_phrases(neighboring_phrases);
//         // Removed the duplicated title words, such as 'data'
//         // neighbouring_words = neighbouring_words.filter(w => ! title_words.includes(w['word']));
//         // Get top 5 neighbouring words
//         dict[title_word] = neighbouring_words.slice(0, 6);
//     }
//     return dict;
// }
